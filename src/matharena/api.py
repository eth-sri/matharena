import anthropic
import re
import os
import time
import base64
import requests
import json
import tempfile
import traceback
from loguru import logger
from tqdm import tqdm
from google import genai
from google.genai import types
from openai import OpenAI, RateLimitError
from together import Together
from anthropic.types import ThinkingBlock, TextBlock
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from matharena.code_execution import CodeRunner, PY_LIBRARIES, EXEC_TIMEOUT

CODE_TOOLS = [{
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": "Executes the code in the given language and returns the standard output and standard error. Your code is always executed as a self-contained script, and it does not have access to the previously executed code blocks! If you use python, your code will be run in an environment with the following libraries installed: " + ", ".join(PY_LIBRARIES),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The self-contained code to execute"
                },
                "lang": {
                    "type": "string",
                    "description": "The programming language of the code (python or cpp)"
                },
            },
            "required": ["code", "lang"],
        }
    }
}]


def encode_image(image_path):
    image_type = image_path.split(".")[-1]
    with open(image_path, "rb") as image_file:
        return image_type, base64.b64encode(image_file.read()).decode("utf-8")

class APIQuery:
    def __init__(self, model, 
                 timeout=6000, 
                 max_tokens=None,
                 api='openai', 
                 max_retries=50,
                 concurrent_requests=30, 
                 is_chat=True,
                 no_system_messages=False,
                 read_cost=1,  
                 write_cost=1,
                 sleep_on_error=60,
                 sleep_after_request=0.1,
                 throw_error_on_failure=False,
                 max_tokens_param="max_tokens",
                 reasoning_effort=None,
                 batch_processing=False,
                 openai_responses=False,
                 n_code_executions=0,
                 **kwargs):
        # if "think" in model and api == "google":
        #     logger.info("Google Think model does not allow chat.")
        #     is_chat = False # think model cannot handle chat
        #     max_tokens_param = "max_output_tokens"
        if api == "google":
            if "thinking_budget" in kwargs.get("config", {}):
                kwargs["config"]["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=kwargs["config"]["thinking_budget"],
                )
                del kwargs["config"]["thinking_budget"]
        if ("o1" in model or "o3" in model or "o4" in model) and api == "openai":
            logger.info("Not using system messages for o1/o3/o4 model.")
            no_system_messages = True # o1 model cannot handle system messages
            if not openai_responses:
                max_tokens_param = "max_completion_tokens"
            if "--" in model:
                model, reasoning_effort = model.split("--")
                logger.info(f"Model: {model}, Reasoning effort: {reasoning_effort}")
        if api not in ["anthropic", "openai"] and batch_processing:
            logger.warning("Batch processing is only supported for the Anthropic API and OpenAI API.")
            batch_processing = False
        if openai_responses and not batch_processing:
            max_tokens_param = "max_output_tokens"

        if n_code_executions > 0 and not openai_responses:
            max_tokens_param = "max_completion_tokens"

        self.kwarg_remover(api, model, kwargs)

        self.model = model
        self.kwargs = kwargs
        if max_tokens is not None:
            self.kwargs[max_tokens_param] = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.throw_error_on_failure = throw_error_on_failure
        self.concurrent_requests = concurrent_requests
        self.is_chat = is_chat
        self.no_system_messages = no_system_messages
        self.sleep_on_error = sleep_on_error
        self.sleep_after_request = sleep_after_request
        self.read_cost = read_cost
        self.write_cost = write_cost
        self.batch_processing = batch_processing
        self.openai_responses = openai_responses
        self.n_code_executions = n_code_executions
        if max_tokens is not None:
            self.max_tokens_param = max_tokens_param
        if reasoning_effort is not None:
            if not self.openai_responses or self.batch_processing:
                self.kwargs["reasoning_effort"] = reasoning_effort
            else:
                self.kwargs["reasoning"] = {"effort": reasoning_effort}

        self.api = api
        self.api_key = None
        self.base_url = None

        self.initialize_api_keys()

    def kwarg_remover(self, api, model, kwargs):
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

    def initialize_api_keys(self):
        if self.api == "xai":
            self.api_key = os.getenv("XAI_API_KEY")
            self.base_url = "https://api.x.ai/v1"
            self.api = "openai"
        elif self.api == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif self.api == "together":
            self.api_key = os.getenv("TOGETHER_API_KEY")
            self.base_url = "https://api.together.xyz/v1"
        elif self.api == "google":
            self.api_key = os.getenv("GOOGLE_API_KEY")
            # if not "think" in self.model:
            #     self.api = "openai"
            #     self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif self.api == "anthropic":
            if self.n_code_executions > 0:
                self.api = "openai"
                self.base_url = "https://api.anthropic.com/v1/"                
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif self.api == "hyperbolic":
            self.api_key = os.getenv("HYPERBOLIC_API_KEY")
            self.base_url = "https://api.hyperbolic.xyz/v1"
            self.api = "openai"
        elif self.api == 'sambanova':
            self.api_key = os.getenv("SAMBA_API_KEY")
            self.base_url = "https://api.sambanova.ai/v1"
            self.api = "openai"
        elif self.api == "deepseek":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.base_url = "https://api.deepseek.com"
            self.api = "openai"
        elif self.api == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
            if "via_openai" in self.kwargs:
                del self.kwargs["via_openai"]
                self.api = "openai"
        elif self.api == "fireworks":
            self.api_key = os.getenv("FIREWORKS_API_KEY")
            self.base_url = "https://api.fireworks.ai/inference/v1"
            self.api = "openai"
        elif self.api == "vllm":
            self.api_key = "token-abc123"
            self.api = "openai"
            self.base_url = f"http://localhost:8000/v1"
            # command = f"vllm serve {self.model} --dtype auto --api-key token-abc123"
            # Launch the command in the back_round.
            # subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Poll the server until it's running.
        else:
            raise ValueError(f"API {self.api} not supported.")

        assert self.api_key is not None, f"API key not found."

    def prepare_query(self, query):
        query, image_path = query
        if not self.is_chat:
            output_query = query[0]["content"]
            for message in query:
                output_query += f"\n\n{'=' * 20}{message['role']}{'=' * 20}\n\n{message['content']}"
            return output_query, image_path
        elif self.no_system_messages:
            # convert system role to user role
            query = [{
                "role": message["role"] if message["role"] != "system" else "user",
                "content": message["content"]
            } for message in query]
        return query, image_path
    
    def get_cost(self, response):
        cost = response["input_tokens"] * self.read_cost + response["output_tokens"] * self.write_cost
        return cost / (10 ** 6)

    def run_queries(self, queries):
        queries_actual = []
        for query in queries:
            if not isinstance(query, tuple):
                queries_actual.append((query, None))
            else:
                queries_actual.append(query)
        if self.api == "vllm":
            while True:
                try:
                    response = requests.get(f"{self.base_url}", timeout=1)
                    if response.status_code == 401: # unauthorized, because no api key here
                        break
                except Exception:
                    pass
                time.sleep(5)
                logger.info("Waiting for VLLM server to start...")
            logger.info("VLLM server started.")

        logger.info(f"Running {len(queries_actual)} queries.")

        if self.batch_processing:
            if self.api == "openai":
                processed_results = self.openai_batch_processing(queries_actual)
            else:
                processed_results = self.anthropic_batch_processing(queries_actual)
            for idx, result in enumerate(processed_results):
                if result is None:
                    result = {
                        "output": "",
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                detailed_cost = {
                    "cost": self.get_cost(result),
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                }
                yield idx, result["output"], detailed_cost
        else:
            with ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
                future_to_index = {
                    executor.submit(self.run_query_with_retry, query): i
                    for i, query in enumerate(queries_actual)
                }
                for future in tqdm(as_completed(future_to_index), total=len(future_to_index)):
                    idx = future_to_index[future]
                    result = future.result()
                    if result is None:
                        result = {
                            "output": "",
                            "input_tokens": 0,
                            "output_tokens": 0,
                        }
                    detailed_cost = {
                        "cost": self.get_cost(result),
                        "input_tokens": result["input_tokens"],
                        "output_tokens": result["output_tokens"],
                    }
                    yield idx, result["output"], detailed_cost
    
    def run_query_with_retry(self, query):
        i = 0
        while i < self.max_retries:
            try:
                output = self.run_query(query)
                time.sleep(self.sleep_after_request)
                return output
            except Exception as e:
                logger.error(f"Error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                time.sleep(self.sleep_on_error)
                # if api error is not due to rate limit, try again
                if "rate limit" not in str(e).lower() and "429" not in str(e):
                    i += 1
                if "violating our usage policy" in str(e).lower():
                    print("Stopping - prompt repeatedly violated usage policy -- ", query)
                    if i > 3:
                        break
                continue
        if self.throw_error_on_failure:
            raise ValueError("Max retries reached.")
        else:
            return {
                "output": "",
                "input_tokens": 0,
                "output_tokens": 0,
            }
    
    def run_query(self, query):
        query = self.prepare_query(query)
        if self.api == "openai":
            if self.n_code_executions > 0:
                return self.openai_query_with_code(query)
            else:
                return self.openai_query(query)
        elif self.api == "together":
            if self.n_code_executions > 0:
                return self.openai_query_with_code(query, is_together=True)
            return self.together_query(query)
        elif self.api == "google":
            if self.n_code_executions > 0:
                return self.google_query_with_code(query)
            else:
                return self.google_query(query)
        elif self.api == "anthropic":
            return self.anthropic_query(query)        
        elif self.api == "openrouter":
            if self.n_code_executions > 0:
                return self.openai_query_with_code(query)
            else:
                return self.openrouter_query(query)
        
    def postprocess_anthropic_result(self, result):
        output_text = ""

        for content in result.content:
            if isinstance(content, ThinkingBlock):
                output_text += "<think>\n"  + content.thinking + "</think>\n\n"
            elif isinstance(content, TextBlock):
                output_text += content.text
                break
        return {
            "output": output_text,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        }

    def anthropic_batch_processing(self, queries, error_repetition=0):
        if error_repetition >= self.max_retries:
            return [
                {
                    "output": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                } for _ in range(len(queries))
            ]

        text_queries = [query[0] for query in queries]
        client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
        )

        requests = []

        for i, text_query in enumerate(text_queries):
            kwargs_here = self.kwargs.copy()
            if text_query[0]["role"] == "system":
                kwargs_here["system"] = text_query[0]["content"]
                text_query = text_query[1:]
            
            request = Request(
                custom_id=f"apiquery-{i}",
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    messages=text_query,
                    **kwargs_here
                )
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
            except:
                logger.warning(f"Error connecting to Anthropic. Retrying in 10s.")
                pass
            if any([current_request_counts[key] != dict(message_batch.request_counts)[key] for key in current_request_counts]):
                current_request_counts = dict(message_batch.request_counts)
                error_sum = sum([current_request_counts[key] for key in current_request_counts if "succeeded" != key])
                logger.info(f"Succeeded Requests Progress: {current_request_counts['succeeded']}/{len(queries)}. Errors: {error_sum}")
            if message_batch.processing_status == "ended":
                break
            time.sleep(10)
        
        outputs = []
        repeat_indices = []

        while True:
            try:
                results = client.messages.batches.results(
                    message_batch_id=message_batch.id,
                )
                break
            except Exception as e:
                logger.error(f"Error connecting to Anthropic: {e}. Retrying in 10 seconds.")
                time.sleep(10)

        for i, result in enumerate(results):  
            if result.result.type == "succeeded":
                outputs.append(self.postprocess_anthropic_result(result.result.message))
            else:
                outputs.append(None)
                repeat_indices.append(i)
                if result.result.type == "errored":
                    logger.error(result.result.error)

        if len(repeat_indices) > 0:
            logger.info(f"Repeating {len(repeat_indices)} queries.")
            repeat_queries = [queries[i] for i in repeat_indices]
            repeat_outputs = self.anthropic_batch_processing(repeat_queries, error_repetition + 1)
            for i, output in zip(repeat_indices, repeat_outputs):
                outputs[i] = output
        
        return outputs
        
    def anthropic_query(self, query):
        query, image_path = query
        client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
            timeout=self.timeout,
        )
        system_message = anthropic.NOT_GIVEN
        if query[0]["role"] == "system":
            system_message = query[0]["content"]
            query = query[1:]
        result = client.messages.create(
            model=self.model,
            messages=query,
            system=system_message,
            **self.kwargs
        )

        return self.postprocess_anthropic_result(result)
    
    def openrouter_query(self, query):
        query, image_path = query
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        query_key = "messages" if self.is_chat else "prompt"

        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions', 
            headers=headers, 
            json={
                'model': self.model,
                query_key: query,
                **self.kwargs
            }
        )
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
            
        json_response = response.json()

        print(json_response)

        if "choices" not in json_response:
            raise Exception(f"Error: {json_response}")

        if self.is_chat:
            output = json_response['choices'][0]['message']['content']
            if "reasoning_content" in json_response['choices'][0]['message'] and json_response['choices'][0]['message']['reasoning_content'] is not None:
                output = json_response['choices'][0]['message']['reasoning_content'] + "</think>\n\n" + output
            return {
                "output": output,
                "input_tokens": json_response['usage']['prompt_tokens'],
                "output_tokens": json_response['usage']['completion_tokens'],
            }
        else:
            output = json_response['choices'][0]['text']
            output = self.skip_repetition(output, query)
            
            reasoning_content = ""
            if "reasoning_content" in json_response['choices'][0] and json_response['choices'][0]['reasoning_content'] is not None:
                reasoning_content = json_response['choices'][0]['reasoning_content']
            if "reasoning" in json_response['choices'][0] and json_response['choices'][0]['reasoning'] is not None:
                reasoning_content = json_response['choices'][0]['reasoning']

            text = "</think>\n\n"
            if len(output) == 0:
                text = ""
            output = reasoning_content + text + output
            
            return {
                "output": output,
                "input_tokens": json_response['usage']['prompt_tokens'],
                "output_tokens": json_response['usage']['completion_tokens'],
            }
    
    def google_query(self, query):
        client = genai.Client(api_key=self.api_key, http_options={'api_version':'v1alpha'})
        query, image_path = query
        parts = []
        if image_path is not None:
            file = client.files.upload(file=image_path)
            assert len(query) == 1
            parts.append(types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type))
        parts.append(types.Part.from_text(text=query[0]["content"]))
        query = [types.Content(role="user", parts=parts)]

        # if "think" in self.model:
        #     config['thinking_config'] = {'include_thoughts': True}
        # config = None
        response = client.models.generate_content(
            model=self.model,
            contents=query,
            **self.kwargs
        )
        output_tokens = response.usage_metadata.total_token_count - response.usage_metadata.prompt_token_count
        # Google API being the Google API...
        assert response.usage_metadata.prompt_token_count is not None
        assert response.usage_metadata.total_token_count is not None
        return {
            "output": "\n\n".join([response.candidates[0].content.parts[i].text 
                                   for i in range(len(response.candidates[0].content.parts))]),
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.total_token_count,
        }

    def together_query(self, query):
        client = Together()
        query, image_path = query
        response = client.chat.completions.create(
            model=self.model,
            messages=query,
            **self.kwargs
        )
        output = response.choices[0].message.content
        if hasattr(response.choices[0].message, "reasoning_content"):
            output = response.choices[0].message.reasoning_content + "\n\n" + output
        return {
            "output": output,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    def openai_batch_processing(self, queries, error_repetition=0):
        if error_repetition >= self.max_retries:
            return [
                {
                    "output": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                } for _ in range(len(queries))
            ]
        text_queries = [query[0] for query in queries]
        jsonl_queries = []

        for i, query in enumerate(text_queries):
            request = {
                "custom_id": f"apiquery-{i}",
                "method": "POST", 
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": query,
                    **self.kwargs
                }
            }
            jsonl_queries.append(request)
        
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, 
                            max_retries=0)
        
        # create temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        with open(tmp.name, "wb") as f:
            for i, query in enumerate(jsonl_queries):
                f.write(json.dumps(query).encode("utf-8"))
                f.write(b"\n")

        batch_input_file = client.files.create(
            file=open(tmp.name, "rb"),
            purpose="batch"
        )

        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        # close tmp file
        tmp.close()

        logger.info(f"Running {len(queries)} queries with batch ID {batch.id} using file with File ID {batch_input_file.id}.")

        request_counts = dict(batch.request_counts)
        
        while True:
            try:
                batch = client.batches.retrieve(batch.id)
            except Exception as e:
                logger.warning(f"Error connecting to OpenAI. Retrying in 10s.")
                pass
            if any([request_counts[key] != dict(batch.request_counts)[key] for key in request_counts]):
                request_counts = dict(batch.request_counts)
                logger.info(f"Completed Requests Progress: {request_counts['completed']}/{len(queries)}. Errors: {request_counts['failed']}/{len(queries)}")
            if batch.status == "completed":
                break
            time.sleep(10)
        
        outputs = [None for _ in range(len(queries))]
        repeat_indices = []

        # if batch.error_file_id is not None:
        #     while True:
        #         try:
        #             error_response = client.files.content(file_id=batch.error_file_id)
        #             break
        #         except Exception as e:
        #             logger.error(f"Error connecting to OpenAI: {e}. Retrying in 10 seconds.")
        #             time.sleep(10)
        #             continue
        #     for line in error_response.iter_lines():
        #         logger.error(line)
        #         json_line = json.loads(line)
        #         if json_line.get("error", 1) is None:
        #             index = int(json_line["custom_id"].split("-")[-1])
        #             outputs[index] = {
        #                 "output": "<Empty response because model reached the maximum output tokens limit.>",
        #                 "input_tokens": 0,
        #                 "output_tokens": 0,
        #             }

        if batch.output_file_id is None:
            return outputs
        while True:
            try:
                file_response = client.files.content(file_id=batch.output_file_id)
                break
            except Exception as e:
                logger.error(f"Error connecting to OpenAI: {e}. Retrying in 10 seconds.")
                time.sleep(10)
                continue

        json_response = []
        for line in file_response.iter_lines():
            json_response.append(json.loads(line))

        for result in json_response:
            index = int(result["custom_id"].split("-")[-1])
            if result["response"]["status_code"] != 200:
                repeat_indices.append(index)
                logger.error(f"Error: {result['response']['status_code']}")
            else:
                try:
                    outputs[index] = {
                        "output": result["response"]["body"]["choices"][0]["message"]["content"],
                        "input_tokens": result["response"]["body"]["usage"]["prompt_tokens"],
                        "output_tokens": result["response"]["body"]["usage"]["completion_tokens"],
                    }
                except Exception as e:
                    logger.error(f"Error: {e}")
                    repeat_indices.append(index)
        
        for i in range(len(outputs)):
            if outputs[i] is None:
                repeat_indices.append(i)
        if len(repeat_indices) > 0:
            logger.info(f"Repeating {len(repeat_indices)} queries.")
            repeat_queries = [queries[i] for i in repeat_indices]
            repeat_outputs = self.openai_batch_processing(repeat_queries, error_repetition + 1)
            for i, output in zip(repeat_indices, repeat_outputs):
                outputs[i] = output
        
        return outputs
    
    def google_query_with_code(self, query):
        function_declarations = [tool["function"] for tool in CODE_TOOLS]        
        tools = types.Tool(function_declarations=function_declarations)        
        config = types.GenerateContentConfig(tools=[tools])
        
        # Extract config parameters and other kwargs
        api_kwargs = self.kwargs.copy()
        if "config" in api_kwargs:
            del api_kwargs["config"]
        
        client = genai.Client(
            api_key=self.api_key,
            # http_options={
            #     "api_version": "v1",
            # }
        )
        
        query, image_path = query
        parts = []
        if image_path is not None:
            file = client.files.upload(file=image_path)
            assert len(query) == 1
            parts.append(types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type))
        parts.append(types.Part.from_text(text=query[0]["content"]))
        query = [types.Content(role="user", parts=parts)]

        response = client.models.generate_content(
            model=self.model,
            config=config,
            contents=query,
            **api_kwargs
        )
        output_contents = []
        output_contents.append(response.candidates[0].content)
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.total_token_count - response.usage_metadata.prompt_token_count
        for it in range(self.n_code_executions):
            print(response)
            if "MALFORMED_FUNCTION_CALL" in str(response.candidates[0].finish_reason) or (response.candidates[0].content.parts is None):
                print("Malformed output, stopping!")
                output_contents.append({"role": "model", "content": "Malformed function call or output, stopping."})
                break
            function_call = None            
            for part in response.candidates[0].content.parts:
                if part.function_call is not None:
                    function_call = part.function_call
                    break
            if function_call is None:
                break
            code = function_call.args["code"]
            lang = function_call.args["lang"]

            code_runner = CodeRunner()
            if lang == "python":
                output = code_runner.execute_python_code(code)
            elif lang == "cpp":
                output = code_runner.execute_cpp_code(code)
            code_runner.terminate()

            if len(output["stdout"]) > 1000:
                output["stdout"] = output["stdout"][:1000] + "\n...<truncated>\n"
            if len(output["stderr"]) > 1000:
                output["stderr"] = output["stderr"][:1000] + "\n...<truncated>\n"
            n_execs_left = self.n_code_executions - it - 1
            info = f"info:\nYou have {n_execs_left} code execution left."
            if output["time"] > EXEC_TIMEOUT:
                info += f"\n\nExecution time exceeded the timeout of {EXEC_TIMEOUT} seconds."
            else:
                info += f"\n\nExecution time: {output['time']} seconds."
            parsed_output = "stdout:\n" + output["stdout"] + "\nstderr:\n" + output["stderr"] + "\n" + info
            function_response_part = types.Part.from_function_response(
                name=function_call.name,
                response={"result": parsed_output},
            )
            output_contents.append(types.Content(role="user", parts=[function_response_part]))
            input_tokens += response.usage_metadata.prompt_token_count
            output_tokens += response.usage_metadata.total_token_count - response.usage_metadata.prompt_token_count
            response = client.models.generate_content(
                model=self.model,
                config=config,
                contents=query + output_contents,
                **api_kwargs
            )
            output_contents.append(response.candidates[0].content)
        parsed_output_msgs = []
        role2role = {"model": "assistant", "user": "user", "tool": "tool", "code": "code"}
        print("output_contents: ", output_contents)
        for content in output_contents:
            print("content: ", content)
            if content is None:
                continue
            if type(content) == dict:
                parsed_output_msgs.append({"role": role2role[content["role"]], "content": content["content"]})
            elif content.role == "user":
                parsed_output_msgs.append({"role": "tool", "content": content.parts[0].function_response.response["result"]})
            elif content.parts is not None:
                for part in content.parts:
                    if part.text is not None:
                        parsed_output_msgs.append({"role": content.role, "content": part.text})
                    if part.function_call is not None:
                        code = part.function_call.args["code"]
                        lang = part.function_call.args["lang"]
                        content = f"""```{lang}\n{code}\n```\n"""
                        parsed_output_msgs.append({"role": "code", "content": content})
                
        print("parsed_output_msgs: ", parsed_output_msgs)
        print("-" * 20)
        print("FINISHED!")
        return {
            "output": parsed_output_msgs,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    
    def openai_query_with_code(self, query, is_together=False):
        if is_together:
            client = Together()
        else:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, 
                            timeout=self.timeout,  max_retries=0)
        messages, image_path = query

        if self.openai_responses:            
            response = None
            max_retries = 5
            while response is None and max_retries > 0:
                try:
                    print("trying messages: ", messages)
                    response = client.responses.create(
                        model=self.model,
                        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
                        input=messages,
                        timeout=self.timeout,
                        **self.kwargs
                    )
                except Exception as e:
                    if "your prompt was flagged" in str(e):
                        messages[0]["content"] = "You are solving a math/coding problem.\n" + messages[0]["content"]
                        max_retries -= 1
                        continue
                    else:
                        raise e
                
            print('response: ', response)
            out_msgs = []
            for out in response.output:
                if out.type == "message":
                    for c in out.content:
                        if c.type == "output_text":
                            out_msgs.append({"role": "assistant", "content": c.text})
                elif out.type == "code_interpreter_call":
                    out_msgs.append({"role": "code", "content": out.code})
                elif out.type == "reasoning":
                    pass
                else:
                    print("output type: ", out.type)
            print('out_msgs: ', out_msgs)
            return {
                "output": out_msgs,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=CODE_TOOLS,
            timeout=self.timeout,
            **self.kwargs
        )
        print(response)
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        output_messages = []
        output_messages.append(response.choices[0].message)
        for it in range(self.n_code_executions):
            if not response.choices[0].message.tool_calls:
                break
            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.function.name == "execute_code":
                    arguments = json.loads(tool_call.function.arguments)
                    code = arguments["code"]
                    lang = arguments["lang"]

                    code_runner = CodeRunner()
                    if lang == "python":
                        output = code_runner.execute_python_code(code)
                    elif lang == "cpp":
                        output = code_runner.execute_cpp_code(code)
                    code_runner.terminate()

                    if len(output["stdout"]) > 1000:
                        output["stdout"] = output["stdout"][:1000] + "\n...<truncated>\n"
                    if len(output["stderr"]) > 1000:
                        output["stderr"] = output["stderr"][:1000] + "\n...<truncated>\n"
                    n_execs_left = self.n_code_executions - it - 1
                    info = f"info:\nYou have {n_execs_left} code execution left."
                    if output["time"] > EXEC_TIMEOUT:
                        info += f"\n\nExecution time exceeded the timeout of {EXEC_TIMEOUT} seconds."
                    else:
                        info += f"\n\nExecution time: {output['time']} seconds."
                    parsed_output = "stdout:\n" + output["stdout"] + "\nstderr:\n" + output["stderr"] + "\n" + info
                    output_messages.append({
                        "role": "tool",
                        "content": parsed_output,
                        "tool_call_id": tool_call.id
                    })
            response = None
            while response is None:
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages + output_messages,
                        tools=None if it == self.n_code_executions - 1 else CODE_TOOLS,
                        timeout=self.timeout,
                        **self.kwargs
                    )
                except Exception as e:
                    logger.info("Got OpenAI error: ", e)                    
                    if isinstance(e, RateLimitError):
                        logger.info("Got OpenAI rate limit error. Sleeping for 60 seconds.")
                        time.sleep(60)
                        continue
                    else:
                        raise e
            input_tokens += response.usage.prompt_tokens
            if "grok" in self.model:
                output_tokens += response.usage.completion_tokens_details.reasoning_tokens + response.usage.completion_tokens
            else:
                output_tokens += response.usage.completion_tokens
            output_messages.append(response.choices[0].message)

        parsed_output_msgs = []
        for msg in output_messages:
            if type(msg) == dict:
                parsed_output_msgs.append({"role": msg["role"], "content": msg["content"]})
                continue
            msg_content = ""
            if hasattr(msg, 'reasoning') and msg.reasoning:
                msg_content += msg.reasoning + "\n"
            if msg.content is not None:
                msg_content = msg.content            
            if len(msg_content) > 0:
                parsed_output_msgs.append({"role": msg.role, "content": msg_content})
            if msg.tool_calls is not None:
                for tool_call in msg.tool_calls:
                    arguments = json.loads(tool_call.function.arguments)
                    code = arguments["code"]
                    lang = arguments["lang"]
                    content = f"""```{lang}\n{code}\n```\n"""
                    parsed_output_msgs.append({"role": "code", "content": content})
        #print("=" * 20)
        #print(parsed_output_msgs)
        #print("input_tokens: ", input_tokens)
        #print("output_tokens: ", output_tokens)

        return {
            "output": parsed_output_msgs,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    
    def openai_query(self, query):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, 
                        timeout=self.timeout, max_retries=0)
        query, image_path = query
        if image_path is not None:
            image_type, base64_image = encode_image(image_path)
            query.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/{image_type};base64,{base64_image}"}}]})

        if not self.openai_responses:
            response = client.chat.completions.create(
                model=self.model,
                messages=query,
                timeout=self.timeout,
                **self.kwargs
            )
            output = response.choices[0].message.content
            if output is None: # in case max token limit reached
                output = ""
            if hasattr(response.choices[0].message, "reasoning_content") and \
                response.choices[0].message.reasoning_content is not None:
                output = response.choices[0].message.reasoning_content + "\n\n" + output
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            if self.base_url is not None and "api.x.ai" in self.base_url:
                output_tokens += response.usage.completion_tokens_details.reasoning_tokens
        else:
            response = client.responses.create(
                model=self.model,
                input=query,
                timeout=self.timeout,
                **self.kwargs
            )
            try:
                output = response.output[-1].content[0].text
            except Exception as e:
                if response.incomplete_details.reason == "max_output_tokens":
                    logger.info("Found incomplete response because of max output tokens. Setting output to the empty string information.")
                    output = "<Empty response because model reached the maximum output tokens limit.>"
                else:
                    raise e
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        return {
            "output": output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    
    def gquery(self, query):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, 
                        timeout=self.timeout, max_retries=0)
        query, image_path = query
        if image_path is not None:
            image_type, base64_image = encode_image(image_path)
            query.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/{image_type};base64,{base64_image}"}}]})

        if not self.openai_responses:
            response = client.chat.completions.create(
                model=self.model,
                messages=query,
                timeout=self.timeout,
                **self.kwargs
            )
            output = response.choices[0].message.content
            if output is None: # in case max token limit reached
                output = ""
            if hasattr(response.choices[0].message, "reasoning_content") and \
                response.choices[0].message.reasoning_content is not None:
                output = response.choices[0].message.reasoning_content + "\n\n" + output
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            if self.base_url is not None and "api.x.ai" in self.base_url:
                output_tokens += response.usage.completion_tokens_details.reasoning_tokens
        else:
            response = client.responses.create(
                model=self.model,
                input=query,
                timeout=self.timeout,
                **self.kwargs
            )
            try:
                output = response.output[-1].content[0].text
            except Exception as e:
                if response.incomplete_details.reason == "max_output_tokens":
                    logger.info("Found incomplete response because of max output tokens. Setting output to the empty string information.")
                    output = "<Empty response because model reached the maximum output tokens limit.>"
                else:
                    raise e
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        return {
            "output": output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }