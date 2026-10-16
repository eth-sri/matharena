"""This module provides a unified API for querying various large language models."""

import inspect
import json
import os
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime

import anthropic
import requests
from anthropic.types import TextBlock, ThinkingBlock
from anthropic.types.beta import BetaTextBlock, BetaThinkingBlock
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming as BetaMessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request as BetaRequest
from loguru import logger
from openai import OpenAI, RateLimitError
from openai.types.responses import Response
from together import Together
from tqdm import tqdm
from transformers import AutoTokenizer

from matharena.request_logger import request_logger
from matharena.token_pricing import TokenPricing
from matharena.utils import check_for_extra_keys

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None


class APIClient:
    """A client that queries various LLM APIs."""

    @staticmethod
    def _parse_apiquery_custom_id(custom_id):
        if not isinstance(custom_id, str) or not custom_id.startswith("apiquery-"):
            raise ValueError(f"Unexpected custom_id format: {custom_id}")
        return int(custom_id.split("-")[-1])

    @staticmethod
    def _lookup_apiquery_local_idx(custom_id, api_idx_to_local_idx):
        api_idx = APIClient._parse_apiquery_custom_id(custom_id)
        if api_idx not in api_idx_to_local_idx:
            raise KeyError(f"Missing custom_id mapping for {custom_id}")
        return api_idx, api_idx_to_local_idx[api_idx]

    def __init__(
        self,
        model,
        timeout=30000,
        max_tokens=None,
        api="openai",
        api_key_env=None,
        base_url=None,
        default_headers=None,
        max_retries=3,
        max_retries_inner=25,
        concurrent_requests=30,
        no_system_messages=False,
        context_limit=None,
        background=False,
        read_cost=1,
        cache_read_cost=None,
        write_cost=1,
        sleep_on_error=60,
        sleep_after_request=0.1,
        include_max_tool_calls=True,
        throw_error_on_failure=False,
        max_tokens_param="max_tokens",
        reasoning_effort=None,
        batch_processing=False,
        batch_max_tokens=None,
        use_openai_responses_api=False,
        use_gdm_tools=False,
        use_google_genai=False,
        stream_openai_chat_completions=False,
        stream_openai_responses=False,
        max_tool_calls=0,
        cache_write_cost=0,
        tools=None,
        anthropic_betas=None,
        long_context_pricing=None,
        **kwargs,
    ):
        """Initializes the APIClient object and params. All prompts are set at run_queries invocation.

        Args:
            model (str): The name of the model to use.
            timeout (int, optional): The timeout for API requests in seconds. Defaults to 30000 (over 8 hours).
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to None.
            api (str, optional): The API to use. Defaults to 'openai'.
            default_headers (dict, optional): Default headers to send with OpenAI-compatible clients.
            max_retries (int, optional): The maximum number of outer attempts for a failed query. Defaults to 3.
            concurrent_requests (int, optional): The number of concurrent requests to make. Defaults to 30.
            no_system_messages (bool, optional): Whether to disable system messages. Defaults to False.
            read_cost (int, optional): The cost of reading a token. Defaults to 1.
            cache_read_cost (int, optional): The cost of reading a cached input token. Defaults to read_cost.
            write_cost (int, optional): The cost of writing a token. Defaults to 1.
            sleep_on_error (int, optional): The number of seconds to sleep on an error. Defaults to 60.
            sleep_after_request (float, optional): The number of seconds to sleep after a request. Defaults to 0.1.
            throw_error_on_failure (bool, optional): Whether to throw an error on failure. Defaults to False.
            max_tokens_param (str, optional): The name of the max_tokens parameter for the API. Defaults to "max_tokens".
            reasoning_effort (str, optional): The reasoning effort to use. Defaults to None.
            batch_processing (bool, optional): Whether to use batch processing. Defaults to False.
            batch_max_tokens (int, optional): Output cap for OpenRouter batches; leaves synchronous max_tokens unchanged.
            use_openai_responses_api (bool, optional): Whether to use OpenAI responses. Defaults to False.
            use_google_genai (bool, optional): Use native Gemini with automatic long-decode continuation.
            stream_openai_chat_completions (bool, optional): Whether to stream OpenAI chat completions internally.
            stream_openai_responses (bool, optional): Whether to stream Responses internally, retaining the final response.
            max_tool_calls (int|dict, optional): The maximum number of tool calls to make. Defaults to 0.
                Could also be a dict that specifies max calls per tool name.
            tools (list, optional): A list of tools to use. Defaults to None.
            long_context_pricing (dict, optional): Higher per-million-token prices for
                requests exceeding input_token_threshold, including cached input.
            **kwargs: Additional keyword arguments for the API.
        """
        # Max tool calls
        self.tool_calls_allowed = False
        if isinstance(max_tool_calls, int):
            self.max_tool_calls = {"any": max_tool_calls}
            self.max_tool_calls_mode = "total"
            if max_tool_calls > 0:
                self.tool_calls_allowed = True
        elif isinstance(max_tool_calls, dict):
            self.max_tool_calls = max_tool_calls
            self.max_tool_calls_mode = "per_tool"
            if sum(max_tool_calls.values()) > 0:
                self.tool_calls_allowed = True

        # Adapt model name and other args to the model
        if "--" in model:
            model, reasoning_effort = model.split("--")
            logger.info(f"Model: {model}, Reasoning effort: {reasoning_effort}")
        if (api not in ["anthropic", "openai", "openrouter"] or self.tool_calls_allowed) and batch_processing:
            logger.warning("Batch processing requires OpenAI, Anthropic, or OpenRouter without tool calling.")
            batch_processing = False
        if ("o1" in model or "o3" in model or "o4" in model or "gpt-5" in model) and api == "openai":
            logger.info("Not using system messages for o1/o3/o4 model.")
            no_system_messages = True  # o1 model cannot handle system messages
            if not use_openai_responses_api:
                max_tokens_param = "max_completion_tokens"
        if use_openai_responses_api and not batch_processing:
            max_tokens_param = "max_output_tokens"
        if self.tool_calls_allowed and not use_openai_responses_api and not api == "anthropic":
            max_tokens_param = "max_completion_tokens"
        self._kwarg_remover(api, model, kwargs)

        self.model = model
        self.kwargs = kwargs
        self.max_tokens_param = max_tokens_param
        self.context_limit = context_limit
        self.max_tokens = max_tokens
        if max_tokens is not None:
            self.kwargs[max_tokens_param] = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retries_inner = max_retries_inner
        self.throw_error_on_failure = throw_error_on_failure
        self.concurrent_requests = concurrent_requests
        self.no_system_messages = no_system_messages
        self.sleep_on_error = sleep_on_error
        self.sleep_after_request = sleep_after_request
        self.read_cost = read_cost
        self.cache_read_cost = read_cost if cache_read_cost is None else cache_read_cost
        self.write_cost = write_cost
        self.pricing = TokenPricing(read_cost, write_cost, cache_read_cost, cache_write_cost, long_context_pricing)
        self.background = background
        self.batch_processing = batch_processing
        self.batch_max_tokens = batch_max_tokens
        if batch_processing and api == "openrouter" and not model.startswith("anthropic/"):
            raise ValueError("OpenRouter batching currently supports Anthropic models only")
        if batch_max_tokens is not None and (not isinstance(batch_max_tokens, int) or batch_max_tokens < 1):
            raise ValueError("batch_max_tokens must be a positive integer")
        self.anthropic_betas = anthropic_betas or []
        self.use_openai_responses_api = use_openai_responses_api
        self.use_gdm_tools = use_gdm_tools
        self.use_google_genai = use_google_genai
        self.use_google_internal_tools = False
        self.stream_openai_chat_completions = stream_openai_chat_completions
        self.stream_openai_responses = stream_openai_responses
        self.include_max_tool_calls = include_max_tool_calls
        self.cache_write_cost = cache_write_cost
        self.background = background
        if max_tokens is not None:
            self.max_tokens_param = max_tokens_param
        if reasoning_effort is not None:
            if not self.use_openai_responses_api or self.batch_processing:
                self.kwargs["reasoning_effort"] = reasoning_effort
            elif "reasoning" in self.kwargs:
                self.kwargs["reasoning"]["effort"] = reasoning_effort
            else:
                self.kwargs["reasoning"] = {"effort": reasoning_effort}

        # Save tools: user should forward all (even if mix of competition-given and scaffold-given)
        self.tools = tools if tools is not None else []
        self.tool_functions = {
            tool_desc["function"]["name"]: func for func, tool_desc in self.tools if "function" in tool_desc
        }
        self.tool_descriptions = [tool_desc for _, tool_desc in self.tools]
        if (not self.tool_calls_allowed or len(self.tool_descriptions) == 0) and "tool_choice" in self.kwargs:
            del self.kwargs["tool_choice"]

        # Prep api
        self.api = api
        self.base_url = base_url
        self.default_headers = default_headers
        self.api_key_env = api_key_env
        self.api_key = None
        self.terminated = False
        self._initialize_api_keys()

        
        # VLLM-specific initialization
        if self.api == "vllm":
            self.vllm_model = None

    def _initialize_vllm(self):
            if LLM is None:
                raise ImportError("vllm is not installed. pip install vllm")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            vllm_args = {}

            for p in ("temperature", "top_p", "max_tokens", "top_k", "repetition_penalty", "presence_penalty"):
                if p in self.kwargs:
                    vllm_args[p] = self.kwargs.pop(p)
            self.sampling_params = SamplingParams(**vllm_args)
            self.vllm_model = LLM(
                model=self.model, tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
            )
            logger.info(f"Loaded local vllm model `{self.model}` with sampling {self.kwargs}")

    def terminate(self):
        """Terminates the APIClient."""
        self.terminated = True

    def _kwarg_remover(self, api, model, kwargs):
        """Removes kwargs that are not supported by the API or model.

        Args:
            api (str): The API to use.
            model (str): The model to use.
            kwargs (dict): The kwargs to clean.
        """
        if "use_openai_responses_api_tools" in kwargs:
            del kwargs["use_openai_responses_api_tools"]
        if any([kw in model for kw in ["o1", "o3", "o4"]]) and "temperature" in kwargs:
            del kwargs["temperature"]
        for kwarg in ["top_p", "top_k", "temperature"]:
            if kwarg in kwargs and kwargs[kwarg] is None:
                del kwargs[kwarg]
        if (api == "anthropic" and "claude-3-7" in model) or (("o1" in model or "o3" in model) and api == "openai"):
            for kwarg_to_remove in ["top_p", "top_k", "temperature"]:
                if kwarg_to_remove in kwargs:
                    logger.info(f"Removing {kwarg_to_remove} parameter for {model} model.")
                    del kwargs[kwarg_to_remove]

    def _initialize_api_keys(self):
        """Initializes the API keys and base URLs for the selected API."""
        if self.api == "sri":
            self.api_key = os.getenv("SRI_API_KEY")
            self.base_url = "https://srlx.inf.ethz.ch/openai"
            self.api = "openai"
        elif self.api == "xai":
            self.api_key = os.getenv("XAI_API_KEY")
            self.base_url = "https://api.x.ai/v1"
            self.api = "openai"
        elif self.api == "stepfun":
            self.api_key = os.getenv("STEPFUN_API_KEY")
            self.base_url = "https://api.stepfun.ai/v1"
            self.api = "openai"
        elif self.api == "insait":
            metadata = json.dumps({
                "project_id": os.environ["CF_PROJECT_ID"],
                "user_id": os.environ["CF_USER_ID"],
            })
            self.api_key = os.getenv("CF_AIG_TOKEN")
            self.base_url = "https://ai-gateway.plain-flower-4887.workers.dev/compat"
            self.default_headers = {"cf-aig-metadata": metadata}
            self.api = "openai"
        elif self.api == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif self.api == "together":
            self.api_key = os.getenv("TOGETHER_API_KEY")
            self.base_url = "https://api.together.xyz/v1"
        elif self.api == "google":
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if self.use_google_genai:
                self.api = "google"
            elif self.tool_calls_allowed and (
                "gdm-eval-model-bcn" in self.model
                or self.use_gdm_tools
            ):
                self.use_google_internal_tools = True
                self.api = "google"
                self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            else:
                self.api = "openai"  # !
                self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif self.api == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif self.api == "glm":
            self.api_key = os.getenv("GLM_API_KEY")
            self.base_url = "https://api.z.ai/api/paas/v4/"
            self.api = "openai"
        elif self.api == "bigmodel":
            self.api_key = os.getenv("BIGMODEL_API_KEY")
            self.base_url = "https://open.bigmodel.cn/api/paas/v4/"
            self.api = "openai"
        elif self.api == "tiiuae":
            self.api_key = os.getenv("TIIUAE_API_KEY")
            self.base_url = "https://falcon-stage.blueoc.tech/v1"
            self.api = "openai"
        elif self.api == "moonshot":
            self.api_key = os.getenv("MOONSHOT_API_KEY")
            self.base_url = "https://api.moonshot.ai/v1"
            self.api = "openai"
        elif self.api == "deepseek":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.base_url = "https://api.deepseek.com"
            self.api = "openai"
        elif self.api == "deepseek_special":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.base_url = "https://api.deepseek.com/v3.2_speciale_expires_on_20251215"
            self.api = "openai"
        elif self.api == "meta":
            self.api_key = (
                os.getenv(self.api_key_env) if self.api_key_env is not None
                else os.getenv("META_API_KEY") or os.getenv("MODEL_API_KEY")
            )
            self.base_url = self.base_url or "https://api.meta.ai/v1"
            self.api = "openai"
        elif self.api == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
            if "via_openai" in self.kwargs:
                del self.kwargs["via_openai"]
                self.api = "openai"
        elif self.api == "custom":
            self.api = "openai"
            self.base_url = self.base_url
            self.api_key = os.getenv(self.api_key_env) if self.api_key_env is not None else "EMPTY"
        elif self.api == "vllm":
            return
        else:
            raise ValueError(f"API {self.api} not supported.")


        assert self.api_key is not None, "API key not found."

    class InternalRequestResult:
        """A class to hold the result of a request internally (below run_queries)."""

        def __init__(self, conversation, input_tokens, output_tokens, cached_input_tokens=0, cached_write_tokens=0, n_retries=0, time=0, cost=None):
            self.conversation = conversation
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.cached_input_tokens = cached_input_tokens
            self.cached_write_tokens = cached_write_tokens
            self.n_retries = n_retries
            self.time = time
            self.cost = cost

    def run_queries(self, queries, no_tqdm=False, ignore_tool_calls=False, custom_indices=None):
        """Only entry point: runs a given list of queries through the API.

        Args:
            queries (list[MessageList]): A list of queries to run. Each query is a MessageList that we will mangle into
            the right format for this API.
            no_tqdm (bool, optional): Whether to disable the tqdm progress bar. Defaults to False.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction. Defaults to False.

        Yields:
            tuple: An (idx, conversation, detailed_cost) tuple.
                idx: Integer index of the query this response corresponds to in [0, len(queries)-1].
                conversation: Full list of messages (including those from the query) in the API format (incl. CoT).
                detailed_cost: A dict with total "cost" ($), "input_tokens", "output_tokens", and "time" (seconds).
                    "request_failed" is True when retries produced no result; this must not be graded as an answer.
        """
        if not no_tqdm:
            logger.info(f"Running {len(queries)} queries.")

        # For now only switches between system/developer, keeps rest intact
        queries = [self._validate_and_prepare_query(query) for query in queries]

        # Prepare batch indices, agents use custom ones for request_logger
        if custom_indices is not None:
            indices = custom_indices
        else:
            indices = list(range(len(queries)))

        # Case 1: VLLM
        if self.api == "vllm":
            # Bypass threading and batch everything into one local generate
            # TODO indices and request logger
            yield from self._run_vllm_queries(queries)
            return

        # Case 2: Batch API
        if self.batch_processing:
            start_time = time.time()
            if self.api == "openai":
                results_batch = self._openai_batch_processing(queries, indices)
            elif self.api == "openrouter":
                results_batch = self._openrouter_batch_processing(queries, indices)
            elif self.anthropic_betas:
                results_batch = self._anthropic_batch_processing_beta(queries, indices)
            else:
                results_batch = self._anthropic_batch_processing(queries, indices)
            end_time = time.time()  # pack all time into first index (since batch)
            for idx, result in enumerate(results_batch):
                request_failed = result is None
                if result is None:
                    conversation = [m.copy() for m in queries[idx]] + [{"role": "assistant", "content": ""}]
                    result = self.InternalRequestResult(conversation, input_tokens=0, output_tokens=0)
                detailed_cost = {
                    "cost": self._result_cost(result, batch=True) * (0.5 if self.api == "openrouter" else 1),
                    "input_tokens": result.input_tokens,
                    "cached_input_tokens": result.cached_input_tokens,
                    "cached_write_tokens": result.cached_write_tokens,
                    "output_tokens": result.output_tokens,
                    "time": end_time - start_time,
                    "n_retries": result.n_retries,
                }
                if request_failed:
                    detailed_cost["request_failed"] = True
                yield idx, result.conversation, detailed_cost
            return

        # Case 3: Standard API; parallelize manually
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
            future_to_index = {
                executor.submit(self._run_query_with_retry, idx, query, ignore_tool_calls): idx
                for idx, query in zip(indices, queries)
            }

            iterator = as_completed(future_to_index)
            if not no_tqdm:
                iterator = tqdm(iterator, total=len(future_to_index))

            for future in iterator:
                idx = future_to_index[future]
                result = future.result()
                request_failed = result is None
                if result is None:
                    conversation = [m.copy() for m in queries[idx]] + [{"role": "assistant", "content": ""}]
                    result = self.InternalRequestResult(conversation, input_tokens=0, output_tokens=0)
                detailed_cost = {
                    "cost": self._result_cost(result),
                    "input_tokens": result.input_tokens,
                    "cached_input_tokens": result.cached_input_tokens,
                    "cached_write_tokens": result.cached_write_tokens,
                    "output_tokens": result.output_tokens,
                    "n_retries": result.n_retries,
                    "time": time.time() - start_time,
                    "request_time": result.time,
                }
                if request_failed:
                    detailed_cost["request_failed"] = True
                yield idx, result.conversation, detailed_cost

    def _validate_and_prepare_query(self, query):
        """Prepares a query for the API.
            All "tool_response" and "assistant" blocks must have come straight from this APIClient
                => We only need to normalize the developer and user messages
            We will assume they arrive in normalized format (only role: "user"/"developer" and content:str fields)

        Args:
            query (MessageList): List of messages to prepare.

        Returns:
            query_prepared (MessageList): The prepared conversation in the format for this API.
        """
        query_prepared = []
        for m in query:
            query_prepared.append(m.copy())
            if m.get("role", "") == "developer" and not self.no_system_messages:
                query_prepared[-1]["role"] = "system"  # use system if expected by API
            if m.get("role", "") == "user":
                check_for_extra_keys(m, ["role", "content", "tool_context"])

            # Fix images into another format for gemini and grok and qwen and glm
            if (
                m.get("role", "") == "user"
                and isinstance(m.get("content", ""), list)
                and (
                    "gemini-" in self.model
                    or "riftrunner" in self.model
                    or "gdm-eval" in self.model
                    or "grok-4" in self.model
                    or "qwen" in self.model
                    or "glm" in self.model
                    or "moonshot" in self.model
                )
            ):
                new_content = []
                for block in m["content"]:
                    if isinstance(block, dict) and block.get("type", "") == "input_image" and "image_url" in block:
                        inner = {"url": block["image_url"]}
                        if "grok-4" in self.model:
                            inner["detail"] = "high"
                        new_content.append({"type": "image_url", "image_url": inner})
                    elif isinstance(block, dict) and block.get("type", "") == "input_text" and "text" in block:
                        new_content.append({"type": "text", "text": block["text"]})
                    elif isinstance(block, dict) and block.get("type", "") in ["text", "image_url"]:
                        # Already transformed (e.g., during last_chance), keep as-is
                        new_content.append(block)
                    else:
                        # Unknown block type, log warning but keep the block to avoid data loss
                        logger.warning(f"Unknown content block type in user message: {block}")
                        new_content.append(block)
                query_prepared[-1]["content"] = new_content

            # Fix images for anthropic
            if m.get("role", "") == "user" and isinstance(m.get("content", ""), list) and self.api == "anthropic":
                new_content = []
                for block in m["content"]:
                    if isinstance(block, dict) and block.get("type", "") == "input_image" and "image_url" in block:
                        b64_full = block["image_url"]
                        tag = "data:image/png;base64,"
                        assert b64_full.startswith(tag)
                        source = {"type": "base64", "media_type": "image/png", "data": b64_full[len(tag) :]}
                        new_content.append({"type": "image", "source": source})
                    elif isinstance(block, dict) and block.get("type", "") == "input_text" and "text" in block:
                        new_content.append({"type": "text", "text": block["text"]})
                    elif isinstance(block, dict) and block.get("type", "") in ["text", "image"]:
                        # Already transformed (e.g., during last_chance), keep as-is
                        new_content.append(block)
                    else:
                        # Unknown block type, log warning but keep the block to avoid data loss
                        logger.warning(f"Unknown content block type in user message: {block}")
                        new_content.append(block)
                query_prepared[-1]["content"] = new_content
        return query_prepared

    def _usage_to_dict(self, usage):
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        return {}

    def _extract_usage_tokens(self, usage):
        usage_dict = self._usage_to_dict(usage)
        input_tokens = usage_dict.get("input_tokens", usage_dict.get("prompt_tokens", 0)) or 0
        output_tokens = usage_dict.get("output_tokens")
        if output_tokens is None:
            completion_tokens = usage_dict.get("completion_tokens")
            if completion_tokens is not None and "total_tokens" not in usage_dict:
                output_tokens = completion_tokens
            else:
                total_tokens = usage_dict.get("total_tokens")
                output_tokens = (total_tokens - input_tokens) if total_tokens is not None else 0

        input_details = usage_dict.get("input_tokens_details", {})
        if not isinstance(input_details, dict):
            input_details = self._usage_to_dict(input_details)
        prompt_details = usage_dict.get("prompt_tokens_details", {})
        if not isinstance(prompt_details, dict):
            prompt_details = self._usage_to_dict(prompt_details)

        cached_input_tokens = (
            usage_dict.get("cached_input_tokens")
            or usage_dict.get("cache_read_input_tokens")
            or input_details.get("cached_tokens")
            or prompt_details.get("cached_tokens")
            or 0
        )
        if usage_dict.get("cache_read_input_tokens"):
            input_tokens += cached_input_tokens  # if cache_read_input_tokens is provided, Anthropic doesn't count those in input_tokens, so we add them back for consistency with other APIs.

        cached_input_tokens = min(max(cached_input_tokens, 0), max(input_tokens, 0))
        cached_creation_input_tokens = usage_dict.get("cache_creation_input_tokens", 0)
        if self.pricing.threshold is not None:
            included_cache_writes = (
                usage_dict.get("cache_write_input_tokens")
                or input_details.get("cache_write_tokens")
                or input_details.get("cache_write_input_tokens")
                or prompt_details.get("cache_write_tokens")
                or prompt_details.get("cache_write_input_tokens")
                or 0
            )
            if included_cache_writes:
                cached_creation_input_tokens = included_cache_writes
                input_tokens = max(0, input_tokens - included_cache_writes)
        return input_tokens, output_tokens, cached_input_tokens, cached_creation_input_tokens

    def _get_cost(self, input_tokens, output_tokens, cached_input_tokens=0, cached_creation_input_tokens=0):
        return self.pricing.cost(input_tokens, output_tokens, cached_input_tokens, cached_creation_input_tokens)

    def _result_cost(self, result, *, batch=False):
        if result.cost is not None:
            return result.cost
        # A batch item/streamed reply is one request. Tool loops price their
        # individual responses before combining token counts.
        cache_writes = result.cached_write_tokens if batch or self.pricing.threshold is not None else 0
        return self._get_cost(result.input_tokens, result.output_tokens, result.cached_input_tokens, cache_writes)

    def _get_messages_from_anthropic_content(self, content):
        """Postprocesses the content from an Anthropic API query.

        Args:
            content: The content from the Anthropic API.

        Returns:
            str: The textual representation.
        """
        messages = []
        for content_block in content:
            if isinstance(content_block, (ThinkingBlock, BetaThinkingBlock)):
                messages.append({"role": "assistant", "type": "cot", "content": content_block.thinking})
            elif isinstance(content_block, (TextBlock, BetaTextBlock)):
                messages.append({"role": "assistant", "type": "response", "content": content_block.text})
                break
        return messages

    def _drop_cot(self, messages):
        """Drops all CoT/thinking/reasoning messages from a conversation.
        This is a cost saving measure at API call site; conversations that are maintained will have this.

        Args:
            messages (MessageList): The conversation to drop CoT from.
        Returns:
            MessageList: The conversation without CoT messages.
        """
        new_messages = []
        for m in messages:
            if m.get("role", "") == "assistant" and m.get("type", "response") in ["cot", "thinking", "reasoning"]:
                continue
            new_messages.append(m.copy())
            new_messages[-1].pop("tool_context", None)

            # Remove thought signatures for riftrunner and bcn
            if m.get("thought_signature", None) is not None:
                new_messages[-1].pop("thought_signature")

            # Remove thought signatures for riftrunner and bcn
            if m.get("extra_content", None) is not None:
                if m["extra_content"].get("google", None) is not None:
                    if m["extra_content"]["google"].get("thought_signature", None) is not None:
                        new_messages[-1]["extra_content"]["google"].pop("thought_signature")

        return new_messages

    def _get_tool_context(self, messages):
        tool_context = {}
        for m in messages:
            if m.get("role", "") == "user" and isinstance(m.get("tool_context"), dict):
                tool_context.update(m["tool_context"])
        return tool_context

    def _execute_tool_function(self, tool_name, arguments, messages):
        tool_func = self.tool_functions[tool_name]
        try:
            signature = inspect.signature(tool_func)
            arguments = arguments.copy()
            for param_name, param in signature.parameters.items():
                if param_name in arguments:
                    continue
                if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                    continue
                if param_name == "messages":
                    arguments["messages"] = messages
                    continue
                tool_context = self._get_tool_context(messages)
                if param_name in tool_context:
                    arguments[param_name] = tool_context[param_name]
        except (TypeError, ValueError):
            pass
        return tool_func(**arguments)

    """
        Case 1: VLLM
    """

    def _run_vllm_queries(self, queries):
        """
        Batch all queries into one vllm.generate(...) call, collect final states, yield in order.

        Args:
            queries (List[str]): A list of queries (chat formatted convos) to run.

        Yields:
            tuple: An (idx, conversation, detailed_cost) tuple.
        """
        if self.vllm_model is None:
            self._initialize_vllm()
        tasks = []
        for idx, query in enumerate(queries):
            query_in_template = self.tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True)
            tasks.append({"id": str(idx), "prompt": query_in_template})

        logger.info(f"Running {len(tasks)} queries on local vllm…")
        last_outputs = []
        time_start = time.time()
        for batch in self.vllm_model.generate(tasks, sampling_params=self.sampling_params):
            for out in batch.outputs:
                last_outputs.append(out)
            if self.terminated:
                last_outputs += [None for _ in range(len(tasks) - len(last_outputs))]
        time_end = time.time()

        for idx, out in enumerate(last_outputs):
            if out is None:
                text = ""
            else:
                text = out.text
            conversation = [m.copy() for m in queries[idx]] + [{"role": "assistant", "content": text}]
            inp = len(self.tokenizer.apply_chat_template(queries[idx], tokenize=True, add_generation_prompt=True,enable_thinking=True)['input_ids'])
            outp = len(out.token_ids)
            detailed_cost = {
                "cost": self._get_cost(inp, outp, cached_input_tokens=0),
                "input_tokens": inp,
                "cached_input_tokens": 0,
                "output_tokens": outp,
                "n_retries": 0,
                "time": time_end - time_start if idx == 0 else 0,  # put all in first since batch
            }
            yield idx, conversation, detailed_cost

    """
        Case 2: Batch API
    """

    def _openrouter_batch_processing(self, queries, indices):
        """Run Anthropic Messages batches through OpenRouter, retrying only failed items.

        A created batch is never resubmitted because polling failed: it may still
        be running. Its ID is recorded in each request log for later retrieval.
        """
        results = [None] * len(queries)
        if not queries:
            return results
        params = self.kwargs.copy()
        extra = dict(params.pop("extra_body", {}))
        provider = extra.pop("provider", {})
        params.update(extra)
        effort = params.pop("reasoning_effort", None)
        if effort is not None:
            params.setdefault("thinking", {"type": "adaptive"})
            params["output_config"] = {**params.get("output_config", {}), "effort": effort}
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            params.pop(key, None)
        params["max_tokens"] = self.batch_max_tokens or self.max_tokens
        if not params["max_tokens"]:
            raise ValueError("OpenRouter Messages batches require max_tokens or batch_max_tokens")

        headers = {**(self.default_headers or {}), "Authorization": f"Bearer {self.api_key}"}
        if self.anthropic_betas:
            headers["anthropic-beta"] = ",".join(self.anthropic_betas)
        url = self.base_url.rstrip("/") + "/batches"
        pending = list(range(len(queries)))
        with requests.Session() as session:
            session.headers.update(headers)
            for retry_idx in range(self.max_retries):
                ts = datetime.now().strftime("%m%d-%H:%M:%S.%f")
                items = []
                for local_idx in pending:
                    api_idx = indices[local_idx]
                    system, messages = self._convert_to_anthropic_messages(queries[local_idx])
                    body = {"messages": messages, **params}
                    if system is not anthropic.NOT_GIVEN:
                        body["system"] = system
                    item = {"custom_id": f"apiquery-{api_idx}", "body": body}
                    items.append(item)
                    request_logger.log_request(
                        ts=ts, batch_idx=api_idx, request=item,
                        model=self.model, endpoint="/v1/messages", provider=provider.get("only"),
                        anthropic_betas=self.anthropic_betas, n_retries=retry_idx,
                    )
                # OpenRouter stream-parses JSON: all batch options must precede requests.
                payload = {"endpoint": "/v1/messages", "model": self.model,
                           "completion_window": "24h"}
                # Batch routing accepts only `only`; sync preferences stay on the sync path.
                if provider.get("only"):
                    payload["provider"] = {"only": provider["only"]}
                payload["requests"] = items
                try:
                    response = session.post(url, json=payload, timeout=self.timeout)
                    response.raise_for_status()
                    batch = response.json()
                    if not isinstance(batch, dict) or not isinstance(batch.get("id"), str):
                        raise ValueError("Invalid OpenRouter batch acknowledgement")
                    batch_id = batch["id"]
                except requests.HTTPError as exc:
                    status = exc.response.status_code
                    logger.error(f"OpenRouter batch submission failed ({status}): {exc.response.text}")
                    for local_idx in pending:
                        request_logger.log_response(ts=ts, batch_idx=indices[local_idx],
                                                    response={"http_status": status, "error": exc.response.text})
                    if status != 429 and status < 500:
                        break
                    if retry_idx + 1 < self.max_retries:
                        time.sleep(self.sleep_on_error)
                    continue
                except (requests.RequestException, ValueError, KeyError) as exc:
                    # A timeout or malformed acknowledgement may hide an accepted batch.
                    logger.error(f"OpenRouter batch submission outcome is unknown; not resubmitting: {exc}")
                    for local_idx in pending:
                        request_logger.log_response(ts=ts, batch_idx=indices[local_idx],
                                                    response={"submission_error": str(exc)})
                    break
                logger.info(f"Running {len(pending)} queries with OpenRouter batch ID {batch_id}")
                for local_idx in pending:
                    request_logger.log_response(ts=ts, batch_idx=indices[local_idx],
                                                batch_id=batch_id, batch_status=batch.get("status"))
                try:
                    batch = self._wait_for_openrouter_batch(session, url, batch)
                except (requests.RequestException, ValueError, KeyError, RuntimeError) as exc:
                    logger.error(f"Could not retrieve OpenRouter batch {batch_id}; not resubmitting it: {exc}")
                    for local_idx in pending:
                        request_logger.log_response(ts=ts, batch_idx=indices[local_idx],
                                                    batch_id=batch_id, polling_error=str(exc))
                    break
                if batch["status"] != "completed":
                    logger.error(f"OpenRouter batch {batch_id} ended as {batch['status']}: {batch.get('error')}")
                    for local_idx in pending:
                        request_logger.log_response(ts=ts, batch_idx=indices[local_idx],
                                                    response=batch, batch_id=batch_id)
                    if batch["status"] == "cancelled":
                        break
                else:
                    index_map = {indices[i]: i for i in pending}
                    seen = set()
                    for item in batch.get("results") or []:
                        try:
                            api_idx, local_idx = self._lookup_apiquery_local_idx(item["custom_id"], index_map)
                            if api_idx in seen:
                                raise ValueError(f"Duplicate custom_id: {item['custom_id']}")
                            seen.add(api_idx)
                        except (KeyError, ValueError, TypeError) as exc:
                            logger.error(f"Unexpected OpenRouter batch result: {exc}")
                            continue
                        request_logger.log_response(ts=ts, batch_idx=api_idx, response=item, batch_id=batch_id)
                        try:
                            reply = item.get("response") or {}
                            if item.get("error") or reply.get("status_code") != 200:
                                raise ValueError(item.get("error") or reply)
                            body = reply["body"]
                            if not isinstance(body, dict) or body.get("error") or body.get("type") != "message":
                                raise ValueError(f"Invalid Messages response: {body}")
                            message = anthropic.types.Message.model_validate(body)
                            if message.stop_reason not in {"end_turn", "stop_sequence", "max_tokens", "refusal"}:
                                raise ValueError(f"Unexpected stop reason: {message.stop_reason}")
                            messages = self._get_messages_from_anthropic_content(message.content)
                            if not any(m.get("type") == "response" for m in messages):
                                messages.append({"role": "assistant", "type": "response", "content": ""})
                            usage = self._extract_usage_tokens(message.usage)
                            results[local_idx] = self.InternalRequestResult(
                                [m.copy() for m in queries[local_idx]] + messages, usage[0], usage[1],
                                cached_input_tokens=usage[2], cached_write_tokens=usage[3], n_retries=retry_idx,
                            )
                        except (KeyError, ValueError, TypeError) as exc:
                            logger.error(f"OpenRouter batch item {item['custom_id']} failed: {exc}")
                pending = [i for i in pending if results[i] is None]
                if not pending:
                    break
                if retry_idx + 1 < self.max_retries:
                    logger.warning(f"Retrying {len(pending)} failed OpenRouter batch items")
                    time.sleep(self.sleep_on_error)
        return results

    def _wait_for_openrouter_batch(self, session, url, batch):
        """Poll one known batch, tolerating initial 404s without creating another."""
        batch_id = batch["id"]
        deadline = time.monotonic() + 24 * 60 * 60 + 5 * 60
        errors = 0
        previous_progress = None
        while batch["status"] not in {"completed", "failed", "cancelled", "expired"}:
            if batch["status"] not in {"validating", "in_progress", "finalizing", "cancelling"}:
                raise ValueError(f"Unknown OpenRouter batch status: {batch['status']}")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Batch {batch_id} exceeded its 24-hour completion window")
            progress = (batch["status"], batch.get("request_counts"))
            if progress != previous_progress:
                previous_progress = progress
                logger.info(f"OpenRouter batch {batch_id}: {progress[0]}, {progress[1]}")
            time.sleep(10)
            try:
                response = session.get(f"{url}/{batch_id}", timeout=self.timeout)
                response.raise_for_status()
                polled = response.json()
                if not isinstance(polled, dict) or "status" not in polled or polled.get("id") != batch_id:
                    raise ValueError("Invalid OpenRouter batch status response")
                batch = polled
                errors = 0
            except (requests.RequestException, ValueError) as exc:
                errors += 1
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in {404, 429} and status < 500:
                    raise
                if errors >= self.max_retries_inner:
                    raise
                logger.warning(f"Error polling OpenRouter batch {batch_id} ({errors}): {exc}")
        return batch

    def _openai_batch_processing(self, queries, indices, retry_idx=0):
        """Processes a batch of queries  using the OpenAI API.

        Args:
            queries (list[MessageList]): A list of queries to run, each is a MessageList in API format.
            retry_idx (int, optional): Current retry index starting from 0.

        Returns:
            list: A list of InternalRequestResult or None.
        """
        if retry_idx >= self.max_retries:
            return [None for _ in range(len(queries))]

        jsonl_queries = []
        for idx, query in zip(indices, queries):
            request = {
                "custom_id": f"apiquery-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": self.model, "messages": self._drop_cot(query), **self.kwargs},
            }
            jsonl_queries.append(request)

        client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, default_headers=self.default_headers, max_retries=0
        )

        # create temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        with open(tmp.name, "wb") as f:
            for i, query in enumerate(jsonl_queries):
                f.write(json.dumps(query).encode("utf-8"))
                f.write(b"\n")

        batch_input_file = client.files.create(file=open(tmp.name, "rb"), purpose="batch")

        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        # close tmp file
        tmp.close()

        logger.info(
            f"Running {len(queries)} queries with batch ID {batch.id} using file with File ID {batch_input_file.id}."
        )

        request_counts = dict(batch.request_counts)

        while True:
            try:
                batch = client.batches.retrieve(batch.id)
            except Exception as e:  # noqa: E722 F841
                logger.warning(f"Error connecting to batch OpenAI. Retrying in 10s. Exception: {e}")
                pass
            if any([request_counts[key] != dict(batch.request_counts)[key] for key in request_counts]):
                request_counts = dict(batch.request_counts)
                logger.info(
                    f"Completed Requests Progress: {request_counts['completed']}/{len(queries)}. Errors: {request_counts['failed']}/{len(queries)}"
                )
            if batch.status == "completed":
                break
            time.sleep(10)

        results = [None for _ in range(len(queries))]
        repeat_indices = []

        if batch.output_file_id is None:
            return results
        while True:
            try:
                file_response = client.files.content(file_id=batch.output_file_id)
                break
            except Exception as e:
                logger.error(f"Error connecting to batch OpenAI. Retrying in 10s. Exception: {e}")
                time.sleep(10)
                continue

        json_response = []
        for line in file_response.iter_lines():
            json_response.append(json.loads(line))

        for result in json_response:
            index = self._parse_apiquery_custom_id(result["custom_id"])
            if result["response"]["status_code"] != 200:
                repeat_indices.append(index)
                logger.error(f"Error: {result['response']['status_code']}")
            else:
                try:
                    content = result["response"]["body"]["choices"][0]["message"]["content"]
                    conversation = [m.copy() for m in queries[index]] + [{"role": "assistant", "content": content}]
                    usage = result["response"]["body"]["usage"]
                    input_tokens, output_tokens, cached_input_tokens, _ = self._extract_usage_tokens(usage)
                    results[index] = self.InternalRequestResult(
                        conversation,
                        input_tokens,
                        output_tokens,
                        cached_input_tokens=cached_input_tokens,
                        n_retries=retry_idx,
                    )
                except Exception as e:
                    logger.error(f"Error when unpacking batch OpenAI response, will repeat. Exception: {e}")
                    repeat_indices.append(index)

        for i in range(len(results)):
            if results[i] is None:
                repeat_indices.append(i)
        if len(repeat_indices) > 0:
            logger.info(f"Repeating {len(repeat_indices)} queries.")
            repeat_queries = [queries[i] for i in repeat_indices]
            repeat_results = self._openai_batch_processing(repeat_queries, retry_idx + 1)
            for i, result in zip(repeat_indices, repeat_results):
                results[i] = result

        return results

    def _anthropic_batch_processing(self, queries, indices, retry_idx=0):
        """Processes a batch of queries using the Anthropic API.

        Args:
            queries (list[MessageList]): A list of queries to run, each is a MessageList in API format.
            retry_idx (int, optional): Current retry index starting from 0.

        Returns:
            list: A list of InternalRequestResult or None.
        """
        if retry_idx >= self.max_retries:
            return [None for _ in range(len(queries))]

        client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
        )

        requests = []
        ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
        ts += f".{datetime.now().microsecond:06d}"
        for idx, query in zip(indices, queries):
            system_message, anthropic_messages = self._convert_to_anthropic_messages(query)
            kwargs_here = self.kwargs.copy()
            if system_message is not anthropic.NOT_GIVEN:
                kwargs_here["system"] = system_message

            payload = {
                "custom_id": f"apiquery-{idx}",
                "params": {"model": self.model, "messages": anthropic_messages, **kwargs_here},
            }
            request_logger.log_request(ts=ts, batch_idx=idx, request=payload)
            request = Request(
                custom_id=f"apiquery-{idx}",
                params=MessageCreateParamsNonStreaming(model=self.model, messages=anthropic_messages, **kwargs_here),
            )
            requests.append(request)

        message_batch = client.messages.batches.create(requests=requests)

        logger.info(f"Running {len(queries)} queries with batch ID {message_batch.id}")

        current_request_counts = dict(message_batch.request_counts)

        while True:
            try:
                message_batch = client.messages.batches.retrieve(
                    message_batch_id=message_batch.id,
                )
            except Exception as e:  # noqa: E722 E841
                logger.warning(f"Error connecting to Anthropic. Retrying in 10s. Exception: {e}")
                pass
            if any(
                [
                    current_request_counts[key] != dict(message_batch.request_counts)[key]
                    for key in current_request_counts
                ]
            ):
                current_request_counts = dict(message_batch.request_counts)
                error_sum = sum([current_request_counts[key] for key in current_request_counts if "succeeded" != key])
                logger.info(
                    f"Succeeded Requests Progress: {current_request_counts['succeeded']}/{len(queries)}. Errors: {error_sum}"
                )
            if message_batch.processing_status == "ended":
                break
            time.sleep(10)

        results = [None for _ in range(len(queries))]
        repeat_indices = []
        api_idx_to_local_idx = {api_idx: local_idx for local_idx, api_idx in enumerate(indices)}

        while True:
            try:
                raw_results = client.messages.batches.results(
                    message_batch_id=message_batch.id,
                )
                break
            except Exception as e:
                logger.error(f"Error connecting to batch Anthropic. Retrying in 10 seconds. Exception: {e}")
                time.sleep(10)

        for raw_result in raw_results:
            try:
                api_idx, local_idx = self._lookup_apiquery_local_idx(raw_result.custom_id, api_idx_to_local_idx)
            except (KeyError, ValueError) as exc:
                logger.error(f"Received unexpected Anthropic batch result custom_id={raw_result.custom_id}: {exc}")
                continue
            request_logger.log_response(ts=ts, batch_idx=api_idx, response=raw_result.model_dump())
            if raw_result.result.type == "succeeded":
                new_messages = self._get_messages_from_anthropic_content(raw_result.result.message.content)
                conversation = [m.copy() for m in queries[local_idx]] + new_messages
                input_tokens, output_tokens, cached_input_tokens, _ = self._extract_usage_tokens(
                    raw_result.result.message.usage
                )
                results[local_idx] = (
                    self.InternalRequestResult(
                        conversation,
                        input_tokens,
                        output_tokens,
                        cached_input_tokens=cached_input_tokens,
                        n_retries=retry_idx,
                    )
                )
            else:
                repeat_indices.append(local_idx)
                if raw_result.result.type == "errored":
                    logger.error(raw_result.result.error)

        for i in range(len(results)):
            if results[i] is None and i not in repeat_indices:
                repeat_indices.append(i)

        if len(repeat_indices) > 0:
            logger.info(f"Repeating {len(repeat_indices)} queries.")
            repeat_queries = [queries[i] for i in repeat_indices]
            repeat_api_indices = [indices[i] for i in repeat_indices]
            repeat_results = self._anthropic_batch_processing(repeat_queries, repeat_api_indices, retry_idx + 1)
            for i, result in zip(repeat_indices, repeat_results):
                results[i] = result

        return results

    def _anthropic_batch_processing_beta(self, queries, indices, retry_idx=0):
        """Processes a batch of queries using the Anthropic beta Message Batches API.

        Mirrors _anthropic_batch_processing but goes through the beta namespace and passes
        self.anthropic_betas (e.g. ["output-300k-2026-03-24"]) so betas such as the 300k-output
        cap are enabled. Routed to from run_queries when anthropic_betas is configured.

        Args:
            queries (list[MessageList]): A list of queries to run, each is a MessageList in API format.
            retry_idx (int, optional): Current retry index starting from 0.

        Returns:
            list: A list of InternalRequestResult or None.
        """
        if retry_idx >= self.max_retries:
            return [None for _ in range(len(queries))]

        client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
        )

        requests = []
        ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
        ts += f".{datetime.now().microsecond:06d}"
        for idx, query in zip(indices, queries):
            system_message, anthropic_messages = self._convert_to_anthropic_messages(query)
            kwargs_here = self.kwargs.copy()
            if system_message is not anthropic.NOT_GIVEN:
                kwargs_here["system"] = system_message

            payload = {
                "custom_id": f"apiquery-{idx}",
                "params": {"model": self.model, "messages": anthropic_messages, **kwargs_here},
            }
            request_logger.log_request(ts=ts, batch_idx=idx, request=payload)
            request = BetaRequest(
                custom_id=f"apiquery-{idx}",
                params=BetaMessageCreateParamsNonStreaming(model=self.model, messages=anthropic_messages, **kwargs_here),
            )
            requests.append(request)

        message_batch = client.beta.messages.batches.create(betas=self.anthropic_betas, requests=requests)

        logger.info(f"Running {len(queries)} queries with batch ID {message_batch.id}")

        current_request_counts = dict(message_batch.request_counts)

        while True:
            try:
                message_batch = client.beta.messages.batches.retrieve(
                    message_batch_id=message_batch.id,
                )
            except Exception as e:  # noqa: E722 E841
                logger.warning(f"Error connecting to Anthropic. Retrying in 10s. Exception: {e}")
                pass
            if any(
                [
                    current_request_counts[key] != dict(message_batch.request_counts)[key]
                    for key in current_request_counts
                ]
            ):
                current_request_counts = dict(message_batch.request_counts)
                error_sum = sum([current_request_counts[key] for key in current_request_counts if "succeeded" != key])
                logger.info(
                    f"Succeeded Requests Progress: {current_request_counts['succeeded']}/{len(queries)}. Errors: {error_sum}"
                )
            if message_batch.processing_status == "ended":
                break
            time.sleep(10)

        results = [None for _ in range(len(queries))]
        repeat_indices = []
        api_idx_to_local_idx = {api_idx: local_idx for local_idx, api_idx in enumerate(indices)}

        while True:
            try:
                raw_results = client.beta.messages.batches.results(
                    message_batch_id=message_batch.id,
                )
                break
            except Exception as e:
                logger.error(f"Error connecting to batch Anthropic. Retrying in 10 seconds. Exception: {e}")
                time.sleep(10)

        for raw_result in raw_results:
            try:
                api_idx, local_idx = self._lookup_apiquery_local_idx(raw_result.custom_id, api_idx_to_local_idx)
            except (KeyError, ValueError) as exc:
                logger.error(f"Received unexpected Anthropic batch result custom_id={raw_result.custom_id}: {exc}")
                continue
            request_logger.log_response(ts=ts, batch_idx=api_idx, response=raw_result.model_dump())
            if raw_result.result.type == "succeeded":
                new_messages = self._get_messages_from_anthropic_content(raw_result.result.message.content)
                conversation = [m.copy() for m in queries[local_idx]] + new_messages
                input_tokens, output_tokens, cached_input_tokens, _ = self._extract_usage_tokens(
                    raw_result.result.message.usage
                )
                results[local_idx] = (
                    self.InternalRequestResult(
                        conversation,
                        input_tokens,
                        output_tokens,
                        cached_input_tokens=cached_input_tokens,
                        n_retries=retry_idx,
                    )
                )
            else:
                repeat_indices.append(local_idx)
                if raw_result.result.type == "errored":
                    logger.error(raw_result.result.error)

        for i in range(len(results)):
            if results[i] is None and i not in repeat_indices:
                repeat_indices.append(i)

        if len(repeat_indices) > 0:
            logger.info(f"Repeating {len(repeat_indices)} queries.")
            repeat_queries = [queries[i] for i in repeat_indices]
            repeat_api_indices = [indices[i] for i in repeat_indices]
            repeat_results = self._anthropic_batch_processing_beta(repeat_queries, repeat_api_indices, retry_idx + 1)
            for i, result in zip(repeat_indices, repeat_results):
                results[i] = result

        return results

    """
        Case 3: Standard API
    """

    def _run_query_with_retry(self, idx, query, ignore_tool_calls=False):
        """Runs a query on standard API with retries on failure.

        Args:
            idx (int): The index of the query in the batch of queries given to run_queries.
            query (MessageList): The query to run.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """
        retry_idx = 0
        total_retries = 0
        start_time = time.time()
        while retry_idx < self.max_retries:
            if self.terminated:
                return None
            try:
                result = self._run_query(idx, query, ignore_tool_calls=ignore_tool_calls)
                result.n_retries += total_retries
                result.time = time.time() - start_time
                time.sleep(self.sleep_after_request)
                return result
            except Exception as e:
                if "Max inner retries reached." in str(e):
                    total_retries += self.max_retries_inner
                elif "rate limit" not in str(e).lower() and "429" not in str(e):
                    total_retries += 1
                logger.error(f"Error in outer retries. Exception: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                time.sleep(self.sleep_on_error)
                # if api error is not due to rate limit, try again
                if "rate limit" not in str(e).lower() and "429" not in str(e):
                    retry_idx += 1
                if "violating our usage policy" in str(e).lower():
                    print("Stopping - prompt repeatedly violated usage policy -- ", query)
                    if retry_idx > 3:
                        break
                continue
        if self.throw_error_on_failure:
            raise ValueError("Max outer retries reached.")
        else:
            return None

    def _run_query(self, idx, query, ignore_tool_calls=False):
        """Runs a query on standard API.

        Args:
            idx (int): The index of the query in the batch of queries given to run_queries.
            query (MessageList): The query to run.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """
        if self.api == "google":
            if self.use_google_genai:
                return self._google_genai_query(idx, query)
            return self._google_query_with_internal_tools(idx, query)
        if self.api == "openai":
            return self._openai_query_with_tools(idx, query, ignore_tool_calls=ignore_tool_calls)
        elif self.api == "together":
            return self._openai_query_with_tools(idx, query, is_together=True, ignore_tool_calls=ignore_tool_calls)
        elif self.api == "anthropic":
            return self._anthropic_query_with_tools(idx, query, ignore_tool_calls=ignore_tool_calls)
        elif self.api == "openrouter":
            return self._openai_query_with_tools(idx, query)

    def _anthropic_query(self, idx, query):
        """Backward-compatible wrapper; Anthropic now uses one unified Messages API path."""
        return self._anthropic_query_with_tools(idx, query, ignore_tool_calls=True)

    def _anthropic_tool_descriptions(self):
        anthropic_tools = []
        for tool_desc in self.tool_descriptions:
            if tool_desc.get("type") == "function" and "function" in tool_desc:
                fn = tool_desc["function"]
                anthropic_tools.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
            elif "name" in tool_desc and "input_schema" in tool_desc:
                anthropic_tools.append(tool_desc)
            else:
                logger.warning(f"Skipping unsupported Anthropic tool descriptor: {tool_desc}")
        return anthropic_tools

    def _to_anthropic_content_blocks(self, content):
        def _safe_text_block(text, cache_control=None):
            if text is None:
                return None
            text_str = str(text)
            if text_str.strip() == "":
                return None
            block = {"type": "text", "text": text_str}
            if cache_control is not None:
                block["cache_control"] = cache_control
            return block

        if isinstance(content, str):
            block = _safe_text_block(content)
            return [] if block is None else [block]
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list):
            block = _safe_text_block(content)
            return [] if block is None else [block]

        out = []
        for block in content:
            if not isinstance(block, dict):
                text_block = _safe_text_block(block)
                if text_block is not None:
                    out.append(text_block)
                continue
            block_type = block.get("type", "")
            if block_type in ["text", "input_text", "output_text"] and "text" in block:
                mapped = _safe_text_block(block.get("text", ""), cache_control=block.get("cache_control", None))
                if mapped is not None:
                    out.append(mapped)
            elif block_type in ["image", "tool_use", "tool_result"]:
                out.append(block)
            else:
                # Preserve known structured blocks; otherwise avoid introducing invalid empty text blocks.
                if "text" in block:
                    mapped = _safe_text_block(block.get("text", ""), cache_control=block.get("cache_control", None))
                    if mapped is not None:
                        out.append(mapped)
                else:
                    out.append(block)
        return out

    def _append_anthropic_message(self, anthropic_messages, role, content):
        content_blocks = self._to_anthropic_content_blocks(content)
        if len(content_blocks) == 0:
            return
        if len(anthropic_messages) > 0 and anthropic_messages[-1]["role"] == role:
            anthropic_messages[-1]["content"].extend(content_blocks)
        else:
            anthropic_messages.append({"role": role, "content": content_blocks})

    def _convert_to_anthropic_messages(self, messages):
        system_message = anthropic.NOT_GIVEN
        anthropic_messages = []

        for m in self._drop_cot(messages):
            role = m.get("role", "")
            if role in ["system", "developer"]:
                if system_message is anthropic.NOT_GIVEN and len(anthropic_messages) == 0:
                    system_blocks = self._to_anthropic_content_blocks(m.get("content", ""))
                    if len(system_blocks) > 0:
                        system_message = system_blocks
                else:
                    self._append_anthropic_message(anthropic_messages, "user", m.get("content", ""))
                continue

            if role == "user":
                self._append_anthropic_message(anthropic_messages, "user", m.get("content", ""))
                continue

            if role == "assistant" and m.get("type", "") in ["function_call", "tool_call"]:
                tool_use_id = m.get("tool_call_id", m.get("id", m.get("call_id", None)))
                if tool_use_id is None:
                    logger.warning(f"Anthropic tool call missing id; skipping block: {m}")
                    continue
                arguments = m.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                self._append_anthropic_message(
                    anthropic_messages,
                    "assistant",
                    [
                        {
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": m.get("tool_name", m.get("name", "")),
                            "input": arguments,
                        }
                    ],
                )
                continue

            if role == "assistant" and m.get("tool_calls", None):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    arguments = fn.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {"raw": arguments}
                    self._append_anthropic_message(
                        anthropic_messages,
                        "assistant",
                        [
                            {
                                "type": "tool_use",
                                "id": tc.get("id"),
                                "name": fn.get("name", ""),
                                "input": arguments,
                            }
                        ],
                    )
                continue

            if role in ["tool", "tool_response", "function_call_output"] or m.get("type", "") == "function_call_output":
                tool_use_id = m.get("tool_call_id", m.get("id", m.get("call_id", None)))
                if tool_use_id is None:
                    logger.warning(f"Anthropic tool response missing tool_call_id; converting to user text: {m}")
                    self._append_anthropic_message(anthropic_messages, "user", m.get("content", m.get("output", "")))
                    continue
                self._append_anthropic_message(
                    anthropic_messages,
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": m.get("content", m.get("output", "")),
                        }
                    ],
                )
                continue

            if role == "assistant":
                if m.get("content", None) is None:
                    continue
                self._append_anthropic_message(anthropic_messages, "assistant", m.get("content", ""))
            elif role != "":
                self._append_anthropic_message(anthropic_messages, "user", m.get("content", ""))

        return system_message, anthropic_messages

    def _anthropic_query_with_tools(self, idx, messages, ignore_tool_calls=False):
        client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
            timeout=self.timeout,
        )

        if ignore_tool_calls:
            max_tool_calls_mode, max_tool_calls = "total", {"any": 0}
        else:
            max_tool_calls_mode, max_tool_calls = self.max_tool_calls_mode, self.max_tool_calls
        total_max_tool_calls = sum(max_tool_calls.values())
        if max_tool_calls_mode == "total":
            nb_executed_tool_calls = {"any": 0}
        else:
            nb_executed_tool_calls = {t: 0 for t in self.tool_functions.keys()}

        conversation = [m.copy() for m in messages]
        system_message, anthropic_messages = self._convert_to_anthropic_messages(conversation)
        anthropic_tools = self._anthropic_tool_descriptions()

        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        cached_write_tokens = 0
        total_retries = 0
        request_cost = 0.0

        while not self.terminated:
            response = None
            n_retries = -1
            while response is None and n_retries < self.max_retries_inner:
                n_retries += 1
                try:
                    payload = {"model": self.model, "messages": anthropic_messages, **self.kwargs}
                    if system_message is not anthropic.NOT_GIVEN:
                        payload["system"] = system_message
                    if len(anthropic_tools) > 0:
                        payload["tools"] = anthropic_tools
                    if ignore_tool_calls:
                        payload["tool_choice"] = {"type": "none"}
                    ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
                    ts += f".{datetime.now().microsecond:06d}"
                    info = {"nb_executed_tool_calls": nb_executed_tool_calls, "n_retries": n_retries}
                    request_logger.log_request(ts=ts, batch_idx=idx, request=payload, **info)
                    response = client.messages.create(**payload)
                    request_logger.log_response(ts=ts, batch_idx=idx, response=response.model_dump())
                except Exception as e:
                    if "rate limit" not in str(e).lower() and "429" not in str(e):
                        total_retries += 1
                    request_logger.log_response(ts=ts, batch_idx=idx, response={"exception": str(e)})
                    logger.error(f"Got Anthropic error in tools inner loop. Exception: {e}")
                    time.sleep(60)
                    continue
            if response is None:
                raise ValueError("Max inner retries reached.")

            step_input, step_output, step_cached_input, step_write_cache = self._extract_usage_tokens(response.usage)
            request_cost += self._get_cost(step_input, step_output, step_cached_input, step_write_cache)
            input_tokens += step_input
            output_tokens += step_output
            cached_input_tokens += step_cached_input
            cached_write_tokens += step_write_cache

            assistant_blocks = []
            text_blocks = []
            tool_uses = []
            for block in response.content:
                block_dict = block.model_dump() if hasattr(block, "model_dump") else block
                if isinstance(block_dict, dict):
                    assistant_blocks.append(block_dict)
                block_type = getattr(block, "type", block_dict.get("type", "") if isinstance(block_dict, dict) else "")
                if block_type == "text":
                    text_blocks.append(getattr(block, "text", block_dict.get("text", "")))
                elif block_type == "thinking":
                    conversation.append({"role": "assistant", "type": "cot", "content": getattr(block, "thinking", "")})
                elif block_type == "tool_use":
                    tool_uses.append(block)

            if len(text_blocks) > 0:
                conversation.append({"role": "assistant", "content": "\n\n".join(text_blocks)})
            if len(assistant_blocks) > 0:
                self._append_anthropic_message(anthropic_messages, "assistant", assistant_blocks)

            if len(tool_uses) == 0:
                break
            if total_max_tool_calls <= sum(nb_executed_tool_calls.values()):
                break

            tool_result_blocks = []
            for tool_use in tool_uses:
                tool_name = getattr(tool_use, "name", "")
                tool_id = getattr(tool_use, "id", None)
                arguments = getattr(tool_use, "input", {})
                output = ""

                if tool_name not in self.tool_functions:
                    output = f"Error: Tool {tool_name} not found."
                    logger.warning(output)
                else:
                    tool_key = "any" if max_tool_calls_mode == "total" else tool_name
                    if nb_executed_tool_calls[tool_key] >= max_tool_calls[tool_key]:
                        output = f"Error: Tool call after exceeding max # of tool calls ({max_tool_calls[tool_key]})."
                    else:
                        try:
                            output = self._execute_tool_function(tool_name, arguments, conversation)
                        except Exception as e:
                            logger.error(f"Error executing tool {tool_name}. Exception: {e}")
                            output = f"Error executing tool {tool_name}. Exception: {e}"
                        if isinstance(output, tuple):
                            output, additional_cost = output
                            input_tokens += additional_cost["input_tokens"]
                            output_tokens += additional_cost["output_tokens"]
                            request_cost += self._get_cost(additional_cost["input_tokens"], additional_cost["output_tokens"])
                        nb_executed_tool_calls[tool_key] += 1

                nb_tool_calls_left = max_tool_calls.get("any", 0) - nb_executed_tool_calls.get("any", 0)
                if max_tool_calls_mode != "total":
                    nb_tool_calls_left = max_tool_calls.get(tool_name, 0) - nb_executed_tool_calls.get(tool_name, 0)
                detail = "for this tool" if max_tool_calls_mode == "per_tool" else "(across all tools)"
                info = f"\n\n### INFO ###\nYou have {nb_tool_calls_left} tool executions left {detail}."
                if not self.include_max_tool_calls:
                    info = ""

                output_str = output if isinstance(output, str) else str(output)
                conversation.append(
                    {
                        "role": "assistant",
                        "type": "function_call",
                        "id": tool_id,
                        "name": tool_name,
                        "arguments": arguments,
                    }
                )
                conversation.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "tool_call_id": tool_id,
                        "content": output_str + info,
                    }
                )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": output_str + info,
                    }
                )

            if len(tool_result_blocks) == 0:
                break
            self._append_anthropic_message(anthropic_messages, "user", tool_result_blocks)

            if total_max_tool_calls <= sum(nb_executed_tool_calls.values()):
                break
        
        if output_tokens == self.max_tokens and (conversation[-1].get("content", "") == "\n\n" or conversation[-1].get("content", "") == "\n"):
            conversation[-1]["content"] = "Model reached its max allowed token limit. \\boxed{None}"
        elif len(conversation) == len(messages):
            conversation.append({"role": "assistant", "content": ""})
        return self.InternalRequestResult(
            conversation,
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
            cached_write_tokens=cached_write_tokens,
            n_retries=total_retries,
            cost=request_cost if self.pricing.threshold is not None else None,
        )

    def _openai_query_with_tools(self, idx, query, is_together=False, ignore_tool_calls=False):
        """Queries the OpenAI API with tools.

        Args:
            idx (int): The index of the query in the batch of queries given to run_queries.
            query (MessageList): The query to run.
            is_together (bool, optional): Whether to use the Together API. Defaults to False.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """
        if is_together:
            client = Together(timeout=self.timeout, max_retries=0)
        else:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=self.default_headers,
                timeout=self.timeout,
                max_retries=0,
            )

        if self.use_openai_responses_api:
            return self._openai_query_responses_api(client, idx, query, ignore_tool_calls=ignore_tool_calls)
        else:
            return self._openai_query_chat_completions_api(client, idx, query, ignore_tool_calls=ignore_tool_calls)

    @staticmethod
    def _normalize_openai_response(response):
        """Normalize compatible-provider dictionaries, including nested output and usage."""
        if isinstance(response, dict):
            return Response.model_construct(**response)
        return response

    @staticmethod
    def _check_responses_error(response):
        """Compatible providers can return a failed Response inside HTTP 200."""
        error = getattr(response, "error", None)
        status = getattr(response, "status", None)
        if error or status in {"failed", "cancelled"}:
            raise ValueError(f"Responses API failed (status={status}): {error}")

    def _openai_query_responses_api(self, client, idx, messages, ignore_tool_calls=False):
        """Queries the OpenAI API with the responses API.

        Args:
            client: The OpenAI client.
            idx (int): The index of the query in the batch of queries given to run_queries.
            messages (list): The messages to send.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """

        # Set up tools
        response_tools = []
        for tool_desc in self.tool_descriptions:
            if tool_desc["type"] != "function":
                response_tools.append(tool_desc)
            else:
                response_tools.append({"type": "function", **tool_desc["function"]})
        if ignore_tool_calls:
            max_tool_calls_mode, max_tool_calls = "total", {"any": 0}
        elif len(response_tools) == 1 and response_tools[0]["type"] == "code_interpreter":
            max_tool_calls_mode, max_tool_calls = "total", {"any": 0}
        else:
            max_tool_calls_mode, max_tool_calls = self.max_tool_calls_mode, self.max_tool_calls

        # State
        total_max_tool_calls = sum(max_tool_calls.values())
        if max_tool_calls_mode == "total":
            nb_executed_tool_calls = {"any": 0}
        else:
            nb_executed_tool_calls = {t: 0 for t in self.tool_functions.keys()}
        conversation = [m.copy() for m in messages]
        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        cached_write_tokens = 0
        total_retries = 0
        request_cost = 0.0

        for _ in range(total_max_tool_calls + 1):
            # Inner retry to get a response
            response = None
            n_retries = -1
            while response is None and n_retries < self.max_retries_inner:
                n_retries += 1
                try:
                    payload = {
                        "model": self.model,
                        "tools": response_tools,
                        "input": self._drop_cot(conversation),  # Drop CoT here to save cost (stays in convo)
                        "timeout": self.timeout,
                        **self.kwargs,
                    }
                    if self.background:
                        payload["background"] = self.background
                    if self.stream_openai_responses:
                        payload["stream"] = True
                    ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
                    ts += f".{datetime.now().microsecond:06d}"
                    info = {"nb_executed_tool_calls": nb_executed_tool_calls, "n_retries": n_retries}
                    request_logger.log_request(ts=ts, batch_idx=idx, request=payload, **info)
                    if self.stream_openai_responses:
                        with client.responses.create(**payload) as events:
                            for event in events:
                                if event.type in {"response.completed", "response.incomplete"}:
                                    # Terminal events include full output and usage, even at the token limit.
                                    response = self._normalize_openai_response(event.response)
                                elif event.type == "response.failed":
                                    failed_response = self._normalize_openai_response(event.response)
                                    request_logger.log_response(ts=ts, batch_idx=idx, response=failed_response.model_dump())
                                    raise ValueError(f"Responses stream failed: {failed_response.error}")
                                elif event.type == "error":
                                    raise ValueError(f"Responses stream error: {event.message}")
                        if response is None:
                            raise ValueError("Responses stream ended without a completed or incomplete response.")
                    else:
                        response = self._normalize_openai_response(client.responses.create(**payload))
                    if self.background:
                        time_start = time.time()
                        while response.status in {"queued", "in_progress"}:
                            time.sleep(60)
                            response = self._normalize_openai_response(client.responses.retrieve(response.id))
                            if time.time() - time_start > self.timeout:
                                raise TimeoutError("Timeout waiting for background response.")
                    request_logger.log_response(ts=ts, batch_idx=idx, response=response.model_dump())
                    self._check_responses_error(response)
                    if self.background and response.usage is None:
                        raise ValueError("No usage info in background response.")
                except Exception as e:
                    if "rate limit" not in str(e).lower() and "429" not in str(e):
                        total_retries += 1
                    request_logger.log_response(
                        ts=ts, batch_idx=idx,
                        exception={"exception": str(e), "traceback": traceback.format_exc()},
                    )
                    time.sleep(60)
                    logger.error(f"Got OpenAI error in responses api inner. Exception: {e}")
                    response = None
                    continue
            if response is None:
                raise ValueError("Max inner retries reached.")

            # Update state: token counts and conversation (potentially execute tool calls)
            step_input, step_output, step_cached_input, step_cache_write = self._extract_usage_tokens(response.usage)
            request_cost += self._get_cost(step_input, step_output, step_cached_input, step_cache_write)
            input_tokens += step_input
            output_tokens += step_output
            cached_input_tokens += step_cached_input
            cached_write_tokens += step_cache_write

            was_tool_call_executed = False
            for out in response.output:
                if out.type == "message":
                    all_messages = ""  # need all together + ID because API crashes otherwise
                    for c in out.content:
                        if c.type == "output_text":
                            all_messages += f"{c.text}\n\n"
                    conversation.append({"role": "assistant", "content": all_messages, "id": out.id})
                elif out.type == "code_interpreter_call":
                    conversation.append(
                        {
                            "type": "code_interpreter_call",
                            "code": out.code,
                            "id": out.id,
                            "container_id": out.container_id,
                        }
                    )
                elif out.type == "web_search_call":
                    conversation.append(
                        {
                            "type": "web_search_call",
                            "query": "\n\n".join(out.action.queries) if out.action.type == "search" else out.action.type,
                            "id": out.id,
                        }
                    )
                elif out.type == "function_call":
                    function_name = out.name
                    arguments = json.loads(out.arguments)
                    tool_func = self.tool_functions[function_name]
                    tool_key = "any" if max_tool_calls_mode == "total" else function_name
                    if nb_executed_tool_calls[tool_key] >= max_tool_calls[tool_key]:
                        output = f"Error: Tool call after exceeding max # of tool calls ({max_tool_calls[tool_key]})."
                    else:
                        try:
                            output = self._execute_tool_function(function_name, arguments, conversation)
                        except Exception as e:
                            logger.error(f"Error executing tool {function_name}. Exception: {e}")
                            output = f"Error executing tool {function_name}. Exception: {e}"
                    if not isinstance(output, str):
                        additional_cost = output[1]
                        input_tokens += additional_cost["input_tokens"]
                        output_tokens += additional_cost["output_tokens"]
                        request_cost += self._get_cost(additional_cost["input_tokens"], additional_cost["output_tokens"])
                        output = output[0]
                    conversation.append(
                        {
                            "type": "function_call",
                            "id": out.id,
                            "call_id": out.call_id,
                            "arguments": out.arguments,
                            "name": out.name,
                        }
                    )
                    was_tool_call_executed = True
                    nb_executed_tool_calls[tool_key] += 1
                    nb_tool_calls_left = max_tool_calls[tool_key] - nb_executed_tool_calls[tool_key]
                    detail = "for this tool" if max_tool_calls_mode == "per_tool" else "(across all tools)"
                    if self.include_max_tool_calls:
                        info = f"\n\n### INFO ###\nYou have {nb_tool_calls_left} tool executions left {detail}."
                    else:
                        info = ""
                    conversation.append(
                        {"type": "function_call_output", "call_id": out.call_id, "output": str(output) + info}
                    )
                elif out.type == "reasoning":
                    """
                    API change - each code_interpreter_call now comes with this block beforehand and if you don't
                    keep it it crashes. So we keep it intact (so it does not get stripped in _drop_cot
                    and clean_conversation later will do what the code below used to do).
                    Still need to convert to dict.
                    """
                    reasoning_block = {
                        "id": out.id,
                        "summary": [{"text": b.text, "type": b.type} for b in out.summary],
                        "type": out.type,
                        "content": [c.model_dump() if hasattr(c, "model_dump") else c for c in out.content] if out.content else out.content,
                        "encrypted_content": out.encrypted_content,
                        # "status": out.status,
                    }
                    # NOTE: you get status from OpenAI but if you send it back to them they crash out
                    conversation.append(reasoning_block)
                    # summary = ""
                    # for thought in out.summary:
                    #    if thought.text is not None:
                    #        summary += "<thought>" + "\n" + thought.text + "\n" + "</thought>\n"
                    # conversation.append({"role": "assistant", "type": "cot", "content": summary, "id": out.id})
                else:
                    raise ValueError(f"Unknown output type {out.type}")

            # If nothing was run this was the last iteration, stop
            if not was_tool_call_executed or self.terminated:
                break
        
        if conversation[-1].get("type", "") == "reasoning":
            raise ValueError("Conversation ended with reasoning block.")
        if len(conversation) == len(messages):
            conversation.append({"role": "assistant", "content": ""})
        return self.InternalRequestResult(
            conversation,
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
            cached_write_tokens=cached_write_tokens,
            n_retries=total_retries,
            cost=request_cost if self.pricing.threshold is not None else None,
        )

    @staticmethod
    def _check_chat_completion_error(response):
        """HTTP 200 can still carry an upstream error, including inside a choice."""
        error = getattr(response, "error", None)
        for choice in getattr(response, "choices", None) or []:
            choice_error = getattr(choice, "error", None)
            if choice_error or getattr(choice, "finish_reason", None) == "error":
                error = choice_error or {"message": "Provider ended the completion with finish_reason=error"}
                break
        if error:
            raise RuntimeError(f"Upstream Chat Completions error: {json.dumps(error, ensure_ascii=False)}")

    def _openai_query_chat_completions_api(self, client, idx, messages, ignore_tool_calls=False):
        """Queries the OpenAI API using chat completions API.

        Args:
            client: The OpenAI client.
            idx (int): The index of the query in the batch of queries given to run_queries.
            messages (list): The messages to send.
            ignore_tool_calls (bool): Whether to ignore tool calls.

        Returns:
            InternalRequestResult or None
        """
        if self.stream_openai_chat_completions:
            return self._openai_query_chat_completions_streaming(client, idx, messages)

        # Set up tools
        if ignore_tool_calls:
            max_tool_calls_mode, max_tool_calls = "total", {"any": 0}
        else:
            max_tool_calls_mode, max_tool_calls = self.max_tool_calls_mode, self.max_tool_calls

        # State
        total_max_tool_calls = sum(max_tool_calls.values())
        if max_tool_calls_mode == "total":
            nb_executed_tool_calls = {"any": 0}
        else:
            nb_executed_tool_calls = {t: 0 for t in self.tool_functions.keys()}
        conversation = [m.copy() for m in messages]
        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        cached_write_tokens = 0
        total_retries = 0
        request_cost = 0.0
        max_output_tokens = self.kwargs.get(self.max_tokens_param, None)

        # As long as we just had a tool response do another request
        was_tool_call_executed = True
        while was_tool_call_executed and not self.terminated:
            was_tool_call_executed = False
            # Inner retry to get a response
            response = None
            n_retries = -1
            while response is None and n_retries < self.max_retries_inner:
                n_retries += 1
                try:
                    kwargs = self.kwargs.copy()
                    kwargs[self.max_tokens_param] = max_output_tokens
                    payload = {
                        "model": self.model,
                        "messages": self._drop_cot(conversation),  # Drop CoT here to save cost (stays in convo)
                        "tools": self.tool_descriptions if len(self.tool_descriptions) > 0 else None,
                        "timeout": self.timeout,
                        **kwargs,
                    }
                    ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
                    ts += f".{datetime.now().microsecond:06d}"
                    info = {"nb_executed_tool_calls": nb_executed_tool_calls, "n_retries": n_retries}
                    request_logger.log_request(ts=ts, batch_idx=idx, request=payload, **info)
                    response = client.chat.completions.create(**payload)
                    request_logger.log_response(ts=ts, batch_idx=idx, response=response.model_dump())
                except Exception as e:
                    if "rate limit" not in str(e).lower() and "429" not in str(e):
                        total_retries += 1
                    request_logger.log_response(ts=ts, batch_idx=idx, response={"exception": str(e)})
                    if isinstance(e, RateLimitError):
                        logger.info(f"Got OpenAI CC rate limit error. Sleeping for 60 seconds. Exception: {e}")
                        time.sleep(60)
                        continue
                    else:
                        if "maximum context length" in str(e).lower() or "input token count" in str(e).lower():
                            max_output_tokens = max_output_tokens // 2
                            logger.info(
                                f"Got OpenAI CC max context length error. Reducing max output tokens to {max_output_tokens} and retrying. Exception: {e}"
                            )
                        logger.info(f"Got OpenAI CC non ratelimit error. Sleeping for 20 seconds: {e}")
                        time.sleep(60)
                        continue
            if response is None:
                raise ValueError("Max inner retries reached.")

            # A provider may fail after HTTP 200, returning partial text plus an error.
            # Keep the raw response log, but let the outer retry discard this attempt.
            self._check_chat_completion_error(response)
            if not response.choices:
                raise RuntimeError("Chat completion returned no choices")

            # Update state: token counts and conversation (potentially execute tool calls)
            step_input, step_output, step_cached_input, step_cache_write = self._extract_usage_tokens(response.usage)
            request_cost += self._get_cost(step_input, step_output, step_cached_input, step_cache_write)
            input_tokens += step_input
            output_tokens += step_output
            cached_input_tokens += step_cached_input
            cached_write_tokens += step_cache_write
            message = response.choices[0].message
            if self.context_limit is not None:
                max_output_tokens = self.context_limit
                if max_output_tokens is not None:
                    max_output_tokens -= response.usage.total_tokens
                    max_output_tokens = min(max_output_tokens, 
                                            self.kwargs.get(self.max_tokens_param, float("inf")))
            else:
                max_output_tokens = self.kwargs.get(self.max_tokens_param, None)
            # Add CoT and rest of message separately
            # TODO: if we notice new ways to return CoT they should be mangled here. Likely missing something.
            has_reasoning_text = False
            if hasattr(message, "reasoning_details") and message.reasoning_details:
                for detail in message.reasoning_details:
                    if isinstance(detail, dict):
                        type_ = detail.get("type", "")
                        text_ = detail.get("text", "")
                        summary_ = detail.get("summary", "")
                    else:
                        type_ = detail.type
                        text_ = getattr(detail, "text", "")
                        summary_ = getattr(detail, "summary", "")
                    if type_ == "reasoning.text":
                        content = text_
                    elif type_ == "reasoning.summary":
                        content = summary_
                    else:
                        continue  # encrypted
                    if content:
                        conversation.append({"role": "assistant", "type": "cot", "content": content})
                        has_reasoning_text = True
            if not has_reasoning_text:
                # Encrypted-only details must not suppress the readable fallback.
                reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
                if reasoning:
                    conversation.append({"role": "assistant", "type": "cot", "content": reasoning})

            message_dict = message.model_dump()
            message_dict = {k: v for k, v in message_dict.items() if v is not None}  # Drop nulls
            conversation.append(message_dict)  # Should have tool calls inside too; and reasoning_details

            # Try to execute all tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if isinstance(tool_call, dict):
                        function_name = tool_call.get("function", dict()).get("name", "")
                        tool_call_id = tool_call.get("id")
                    else:
                        function_name = tool_call.function.name
                        tool_call_id = tool_call.id
                    if function_name not in self.tool_functions:
                        logger.warning(f"Tool {function_name} not found.")
                        conversation.append(
                            {"role": "tool", "tool_name": function_name, "tool_call_id": tool_call_id, "content": "Error: Tool not found."}
                        )
                        continue

                    # If no budget return error
                    # NOTE: just erroring out here might stop the request loop but the model will be given last chance.
                    tool_key = "any" if max_tool_calls_mode == "total" else function_name
                    if nb_executed_tool_calls[tool_key] >= max_tool_calls[tool_key]:
                        error = f"Error: Exceeded maximum number of tool calls ({max_tool_calls[tool_key]})."
                        conversation.append({"role": "tool", "tool_call_id": tool_call.id, "content": error})
                    else:
                        # Execute tool
                        arguments = json.loads(tool_call.function.arguments)
                        try:
                            output = self._execute_tool_function(function_name, arguments, conversation)
                        except Exception as e:
                            logger.error(f"Error executing tool {function_name}. Exception: {e}")
                            output = f"Error executing tool {function_name}. Exception: {e}"
                        # Tools can return additional cost
                        if isinstance(output, tuple):
                            output, extra_cost = output
                            input_tokens += extra_cost["input_tokens"]
                            output_tokens += extra_cost["output_tokens"]
                            request_cost += self._get_cost(extra_cost["input_tokens"], extra_cost["output_tokens"])

                        nb_executed_tool_calls[tool_key] += 1

                        nb_tool_calls_left = max_tool_calls[tool_key] - nb_executed_tool_calls[tool_key]
                        detail = "for this tool" if max_tool_calls_mode == "per_tool" else "(across all tools)"
                        info = f"\n\n### INFO ###\nYou have {nb_tool_calls_left} tool executions left {detail}."
                        conversation.append(
                            {
                                "role": "tool",
                                "tool_name": function_name,
                                "tool_call_id": tool_call.id,
                                "content": str(output) + info,
                            }
                        )

                    was_tool_call_executed = True

                if total_max_tool_calls <= sum(nb_executed_tool_calls.values()):
                    if conversation[-1]["role"] != "assistant":
                        conversation.append({"role": "assistant", "content": "Assistant executed more tools than allowed. \\boxed{None}"})
                    break  # No more tool calls allowed, stop here

        if total_max_tool_calls > 0:
            logger.info(f"Finished on a loop without tool calls, after executing {nb_executed_tool_calls} calls total.")

        return self.InternalRequestResult(
            self.clean_cot_from_conversation(conversation),
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
            cached_write_tokens=cached_write_tokens,
            n_retries=total_retries,
            cost=request_cost if self.pricing.threshold is not None else None,
        )

    def _openai_query_chat_completions_streaming(self, client, idx, messages):
        """Queries the OpenAI API using chat completions API with streaming.

        Tool calls are ignored in streaming mode; we only aggregate the final text.
        """
        conversation = [m.copy() for m in messages]
        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        cached_write_tokens = 0
        total_retries = 0
        max_output_tokens = self.kwargs.get(self.max_tokens_param, None)

        response = None
        n_retries = -1
        while response is None and n_retries < self.max_retries_inner:
            n_retries += 1
            try:
                kwargs = self.kwargs.copy()
                kwargs[self.max_tokens_param] = max_output_tokens
                payload = {
                    "model": self.model,
                    "messages": self._drop_cot(conversation),
                    "timeout": self.timeout,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    **kwargs,
                }
                ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
                ts += f".{datetime.now().microsecond:06d}"
                request_logger.log_request(ts=ts, batch_idx=idx, request=payload, n_retries=n_retries)
                response = client.chat.completions.create(**payload)
            except Exception as e:
                if "rate limit" not in str(e).lower() and "429" not in str(e):
                    total_retries += 1
                request_logger.log_response(ts=ts, batch_idx=idx, response={"exception": str(e)})
                if isinstance(e, RateLimitError):
                    logger.info(f"Got OpenAI CC rate limit error. Sleeping for 60 seconds. Exception: {e}")
                    time.sleep(60)
                    continue
                if "maximum context length" in str(e).lower() or "input token count" in str(e).lower():
                    if max_output_tokens is not None:
                        max_output_tokens = max_output_tokens // 2
                        logger.info(
                            f"Got OpenAI CC max context length error. Reducing max output tokens to {max_output_tokens} and retrying. Exception: {e}"
                        )
                logger.info(f"Got OpenAI CC non ratelimit error. Sleeping for 20 seconds: {e}")
                time.sleep(60)
                continue
        if response is None:
            raise ValueError("Max inner retries reached.")

        text_parts = []
        reasoning_parts = []
        final_usage = None
        generation_id = None
        provider = None
        finish_reason = None
        try:
            with closing(response):
                for event in response:
                    generation_id = getattr(event, "id", None) or generation_id
                    provider = getattr(event, "provider", None) or provider
                    if getattr(event, "usage", None) is not None:
                        final_usage = event.usage
                    self._check_chat_completion_error(event)
                    if not getattr(event, "choices", None):
                        continue
                    choice = event.choices[0]
                    finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                    delta = choice.delta
                    if getattr(delta, "content", None):
                        text_parts.append(delta.content)
                    reasoning_delta = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
                    if reasoning_delta:
                        reasoning_parts.append(reasoning_delta)
            if finish_reason is None:
                raise RuntimeError("Chat completion stream ended without a finish reason (incomplete response)")
        except Exception as e:
            # The SDK raises APIError while iterating an SSE error event, after
            # create() has succeeded. Preserve diagnostics before the outer retry.
            error_body = getattr(e, "body", None)
            request_logger.log_response(
                ts=ts,
                batch_idx=idx,
                response={
                    "streamed": True,
                    "exception": str(e),
                    "error": error_body,
                    "id": generation_id,
                    "provider": provider,
                    "finish_reason": finish_reason,
                    "partial_content": "".join(text_parts),
                    "partial_reasoning": "".join(reasoning_parts),
                    "usage": self._usage_to_dict(final_usage) if final_usage is not None else None,
                },
            )
            logger.warning(
                "Chat stream failed: model={}, batch_idx={}, generation_id={}, provider={}, "
                "content_chars={}, reasoning_chars={}, error={}, error_body={}",
                self.model, idx, generation_id, provider,
                sum(map(len, text_parts)), sum(map(len, reasoning_parts)), str(e), error_body,
            )
            raise

        reasoning = "".join(reasoning_parts)
        if reasoning:
            conversation.append({"role": "assistant", "type": "cot", "content": reasoning})
        content = "".join(text_parts)
        conversation.append({"role": "assistant", "content": content})

        if final_usage is not None:
            input_tokens, output_tokens, cached_input_tokens, cached_write_tokens = self._extract_usage_tokens(final_usage)

        request_logger.log_response(
            ts=ts,
            batch_idx=idx,
            response={
                "streamed": True,
                "id": generation_id,
                "provider": provider,
                "finish_reason": finish_reason,
                "content": content,
                "reasoning": reasoning,
                "usage": final_usage.model_dump() if final_usage is not None else None,
            },
        )

        return self.InternalRequestResult(
            conversation,
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
            cached_write_tokens=cached_write_tokens,
            n_retries=total_retries,
        )

    def clean_cot_from_conversation(self, conversation):
        """Cleans CoT from conversation for cost saving purposes.

        Args:
            conversation (list): The conversation to clean.
        Returns:
            list: The cleaned conversation.
        """
        cleaned_conversation = []
        for message in conversation:
            new_message = message.copy()
            if "reasoning_details" in new_message:
                del new_message["reasoning_details"]
            if "reasoning" in new_message:
                del new_message["reasoning"]
            if "reasoning_content" in new_message:
                del new_message["reasoning_content"]
            cleaned_conversation.append(new_message)
        return cleaned_conversation

    def _google_genai_query(self, idx, messages):
        """Use the native SDK to continue long decodes and retain cumulative usage."""
        from google import genai

        if self.tool_calls_allowed or any(not isinstance(m.get("content", ""), str) for m in messages):
            raise ValueError("Native Google GenAI queries currently support text without tools")
        contents = [
            {"role": "model" if m["role"] == "assistant" else m["role"],
             "parts": [{"text": m.get("content", "")}]} for m in self._drop_cot(messages)
            if m["role"] not in {"system", "developer"}
        ]
        config = {
            "max_output_tokens": self.max_tokens,
            "thinking_config": {"thinking_level": self.kwargs.get("reasoning_effort"),
                                "include_thoughts": True},
            "automatic_continuation": True,
            "continuation_token": self.kwargs.get("continuation_token"),
            "system_instruction": "\n\n".join(
                m["content"] for m in messages if m["role"] in {"system", "developer"}
            ) or None,
        }
        payload = {"model": self.model, "contents": contents, "config": config}
        responses = []
        with genai.Client(api_key=self.api_key, http_options={"timeout": self.timeout * 1000}) as client:
            while True:
                ts = datetime.now().strftime("%m%d-%H:%M:%S.%f")
                request_logger.log_request(ts=ts, batch_idx=idx, request={
                    **payload, "config": genai.types.GenerateContentConfig(**config).model_dump(mode="json")
                })
                response = client.models.generate_content(**payload)
                request_logger.log_response(ts=ts, batch_idx=idx, response=response.model_dump(mode="json"))
                if not response.candidates:
                    raise ValueError("Gemini returned no candidates")
                responses.append(response)
                candidate = response.candidates[0]
                if candidate.finish_reason != "CONTINUATION":
                    break
                if not candidate.continuation_token:
                    raise ValueError("Gemini continuation has no resume token")
                # An empty upstream hop can make the SDK return an intermediate merge.
                config["continuation_token"] = candidate.continuation_token
        parts = [p for r in responses for p in ((r.candidates[0].content.parts or [])
                 if r.candidates[0].content else [])]
        conversation = [m.copy() for m in messages]
        for thought, kind in ((True, "cot"), (False, "response")):
            text = "".join(p.text for p in parts if p.text and bool(p.thought) == thought)
            if text or not thought:
                conversation.append({"role": "assistant", "type": kind, "content": text})
        usages = [r.usage_metadata for r in responses]
        return self.InternalRequestResult(
            conversation, input_tokens=sum(u.prompt_token_count or 0 for u in usages),
            output_tokens=sum((u.candidates_token_count or 0) + (u.thoughts_token_count or 0) for u in usages),
            cached_input_tokens=sum(u.cached_content_token_count or 0 for u in usages),
        )

    def _google_query_with_internal_tools(self, idx, messages):
        """Queries Google for BCN.
        InternalRequestResult or None
        """
        # NOTE: expect single turn (since internal tool calls) and don't reprompt
        assert len(messages) == 1 and messages[0]["role"] == "user"

        conversation = [m.copy() for m in messages]

        headers = {
            "x-goog-api-key": f"{self.api_key}",
            "Content-Type": "application/json",
        }

        function_declarations = []
        google_tools = []
        has_web_search = False
        for tool_desc in self.tool_descriptions:
            if tool_desc.get("type") == "function" and "function" in tool_desc:
                fn = tool_desc["function"]
                function_declarations.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
            elif tool_desc.get("type") == "web_search":
                google_tools.append({"googleSearch": {}})
                has_web_search = True
            elif "google_search" in tool_desc:
                google_tools.append({"googleSearch": tool_desc["google_search"]})
                has_web_search = True
            else:
                google_tools.append(tool_desc)
                if "googleSearch" in tool_desc:
                    has_web_search = True
        if function_declarations:
            google_tools.append({"functionDeclarations": function_declarations})

        payload = {
            "contents": [{"role": "user", "parts": [{"text": messages[0]["content"]}]}],
        }
        if google_tools:
            payload["tools"] = google_tools

        ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
        ts += f".{datetime.now().microsecond:06d}"
        request_logger.log_request(ts=ts, batch_idx=idx, request=payload)

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
        )
        request_logger.log_response(ts=ts, batch_idx=idx, response=response.json())

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        json_response = response.json()

        if "candidates" not in json_response:
            raise Exception(f"Error: {json_response}")

        candidate = json_response["candidates"][0]
        message = candidate["content"]
        parts = message["parts"]
        role = message["role"]
        assert role == "model"

        input_tokens = json_response["usageMetadata"]["promptTokenCount"]
        output_tokens = json_response["usageMetadata"]["candidatesTokenCount"]

        def _clear_buffer(conversation, buffer, mode):
            if buffer is None or len(buffer) == 0:
                return
            if mode == "cot":
                conversation.append({"role": "assistant", "type": "cot", "content": buffer})
            elif mode == "response":
                conversation.append({"role": "assistant", "type": "response", "content": buffer})
            else:
                return

        mode = None
        buffer = ""
        if has_web_search and candidate.get("groundingMetadata"):
            conversation.append({"role": "assistant", "type": "web_search_call", "query": messages[0]["content"]})
        for part in parts:
            if "thought" in part and part["thought"]:
                if mode != "cot":
                    _clear_buffer(conversation, buffer, mode)
                    buffer = ""
                    mode = "cot"
                buffer += f"{part['text']}"
            elif "functionCall" in part:
                _clear_buffer(conversation, buffer, mode)
                buffer = ""
                mode = None
                conversation.append(
                    {
                        "type": "tool_call",
                        "role": "assistant",
                        "call_id": "1",
                        "arguments": part["functionCall"]["args"],
                        "name": part["functionCall"]["name"],
                    }
                )
            elif "functionResponse" in part:
                _clear_buffer(conversation, buffer, mode)
                buffer = ""
                mode = None
                name = part["functionResponse"]["name"]
                resp = part["functionResponse"]["response"]
                if "result" in resp:
                    result = resp["result"]
                elif "content" in resp:
                    result = resp["content"]
                else:
                    result = ""
                conversation.append({"role": "tool", "tool_name": f"{name}", "tool_call_id": "1", "content": result})
            elif "text" in part:
                if mode != "response":
                    _clear_buffer(conversation, buffer, mode)
                    buffer = ""
                    mode = "response"
                buffer += f"{part['text']}"
        _clear_buffer(conversation, buffer, mode)

        return self.InternalRequestResult(
            conversation=conversation, input_tokens=input_tokens, output_tokens=output_tokens
        )
