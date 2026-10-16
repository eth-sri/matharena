"""Exercise provider failures through the real SDK without making API requests."""

import json

import httpx
import pytest
from openai import OpenAI

import matharena.api_client as api_module
from matharena.api_client import APIClient
from matharena.request_logger import RequestLogger


USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 30,
    "total_tokens": 130,
    "prompt_tokens_details": {"cached_tokens": 20},
    "completion_tokens_details": {"reasoning_tokens": 10},
}
ERROR = {
    "code": 502,
    "message": "JSON error injected into SSE stream",
    "metadata": {"error_type": "provider_unavailable", "provider_name": "Z.AI"},
}


def chunk(delta=None, finish_reason=None, **extra):
    return {
        "id": "gen-test", "object": "chat.completion.chunk", "created": 0,
        "model": "z-ai/glm-5.3", "provider": "Z.AI",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
        **extra,
    }


def sse(*events):
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                          stream=httpx.ByteStream(body.encode()))


def successful_stream(finish_reason="stop", reasoning_field="reasoning"):
    return sse(chunk({reasoning_field: "Fresh reasoning."}),
               chunk({"content": "42"}, finish_reason),
               chunk(choices=[], usage=USAGE))


def completion(**extra):
    return {
        "id": "gen-test", "object": "chat.completion", "created": 0,
        "model": "z-ai/glm-5.3-flash", "provider": "Z.AI", "usage": USAGE,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "42",
        }}],
        **extra,
    }


@pytest.fixture
def run_query(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setattr(api_module.time, "sleep", lambda _: None)
    request_log = RequestLogger()
    request_log.enabled = True
    request_log.log_dir = str(tmp_path)
    monkeypatch.setattr(api_module, "request_logger", request_log)
    clients = []

    def run(responses, stream=True, generator_stream=False):
        requests = []

        def handler(request):
            assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
            body = json.loads(request.content)
            requests.append(body)
            assert body["provider"] == {
                "only": ["z-ai"], "allow_fallbacks": False,
            }
            return responses[len(requests) - 1]

        def factory(**kwargs):
            client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
            clients.append(client)
            if generator_stream:
                original_create = client.chat.completions.create

                def create(**payload):
                    with original_create(**payload) as response:
                        yield from response

                monkeypatch.setattr(client.chat.completions, "create", create)
            return client

        monkeypatch.setattr(api_module, "OpenAI", factory)
        client = APIClient(
            model="z-ai/glm-5.3" if stream else "z-ai/glm-5.3-flash", api="openrouter",
            max_tokens=512, stream_openai_chat_completions=stream,
            max_retries=2, max_retries_inner=0, sleep_after_request=0, sleep_on_error=0,
            extra_body={"provider": {"only": ["z-ai"], "allow_fallbacks": False}},
        )
        result = client._run_query_with_retry(0, [{"role": "user", "content": "What is 6*7?"}])
        logs = [json.loads(path.read_text()) for path in sorted(tmp_path.glob("uninitialized/*.json"))]
        assert all(response.is_closed for response in responses[:len(requests)])
        return result, logs, requests

    yield run
    for client in clients:
        client.close()


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
def test_successful_stream_retains_usage_and_legitimate_token_limit(run_query, finish_reason, reasoning_field):
    result, logs, requests = run_query([successful_stream(finish_reason, reasoning_field)])
    assert result.conversation[-2:] == [
        {"role": "assistant", "type": "cot", "content": "Fresh reasoning."},
        {"role": "assistant", "content": "42"},
    ]
    assert (result.input_tokens, result.output_tokens, result.cached_input_tokens) == (100, 30, 20)
    assert result.n_retries == 0
    assert len(requests) == 1
    assert logs[0]["response"]["id"] == "gen-test"
    assert logs[0]["response"]["provider"] == "Z.AI"
    assert logs[0]["response"]["finish_reason"] == finish_reason


@pytest.mark.parametrize("message", ["JSON error injected into SSE stream", "Upstream idle timeout exceeded"])
@pytest.mark.parametrize("has_partial_output", [False, True])
def test_sse_errors_are_logged_and_retried_without_partial_answer(run_query, message, has_partial_output):
    error = {**ERROR, "message": message}
    events = [chunk({"reasoning": "Abandoned reasoning.", "content": "partial answer"})] if has_partial_output else []
    result, logs, requests = run_query([sse(*events, {"error": error}), successful_stream()])
    assert len(requests) == 2
    assert result.n_retries == 1
    assert result.conversation[-1]["content"] == "42"
    assert "Abandoned" not in str(result.conversation)
    assert "partial answer" not in str(result.conversation)
    failed = logs[0]["response"]
    assert failed["error"] == error
    assert failed["exception"] == message
    assert failed["partial_content"] == ("partial answer" if has_partial_output else "")
    assert failed["partial_reasoning"] == ("Abandoned reasoning." if has_partial_output else "")
    assert failed["usage"] is None
    assert "content" not in failed


@pytest.mark.parametrize("events", [[], [chunk({"content": "partial answer"})], [chunk({}, "error")]])
def test_incomplete_or_error_stream_is_not_accepted(run_query, events):
    result, logs, requests = run_query([sse(*events), successful_stream()])
    assert len(requests) == 2
    assert result.n_retries == 1
    assert result.conversation[-1]["content"] == "42"
    assert "exception" in logs[0]["response"]


@pytest.mark.parametrize("error_shape", ["top_level", "choice", "finish_only", "no_choices"])
def test_http_200_provider_error_is_not_an_answer(run_query, error_shape):
    bad = completion()
    if error_shape == "top_level":
        bad.update(error=ERROR, choices=[])
    elif error_shape == "no_choices":
        bad["choices"] = []
    else:
        bad["choices"][0].update(finish_reason="error", message={
            "role": "assistant", "content": None, "reasoning": "Abandoned reasoning.",
        })
        if error_shape == "choice":
            bad["choices"][0]["error"] = ERROR
    result, logs, requests = run_query(
        [httpx.Response(200, json=bad), httpx.Response(200, json=completion())], stream=False,
    )
    assert len(requests) == 2
    assert result.n_retries == 1
    assert result.conversation[-1]["content"] == "42"
    assert "Abandoned" not in str(result.conversation)
    assert logs[0]["response"]["id"] == bad["id"]
    assert (result.input_tokens, result.output_tokens, result.cached_input_tokens) == (100, 30, 20)


def test_persistent_stream_failure_exhausts_retry_budget(run_query):
    result, logs, requests = run_query([sse({"error": ERROR}), sse({"error": ERROR})])
    assert result is None
    assert len(requests) == 2
    assert len(logs) == 2
    assert all(log["response"]["error"] == ERROR for log in logs)


def test_stream_can_be_a_generator_without_context_manager(run_query):
    result, logs, requests = run_query([successful_stream()], generator_stream=True)
    assert result.conversation[-1]["content"] == "42"
    assert len(requests) == 1
