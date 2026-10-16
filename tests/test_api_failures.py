"""Regression coverage for Grok's failed Responses replies and reasoning summaries."""

import json
from pathlib import Path

import httpx
import pytest

import matharena.api_client as api_module
from matharena.api_client import APIClient
from matharena.json_zst import load_json_zst
from matharena.runner import Runner
from matharena.runs import Runs
from matharena.solvers import PureModelSolver, SolverResponse
from matharena.utils import normalize_conversation
from test_meta_responses import install_transport, response_body, sse_response


@pytest.fixture(autouse=True)
def isolated_api(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setattr(api_module.request_logger, "enabled", False)
    monkeypatch.setattr(api_module.time, "sleep", lambda _: None)

    def forbid_network(*args, **kwargs):
        raise AssertionError("Tests must use MockTransport, never the network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbid_network)


def client_args(**overrides):
    return dict(model="x-ai/grok-4.7", api="openrouter", use_openai_responses_api=True,
                max_retries=2, max_retries_inner=1, sleep_on_error=0,
                sleep_after_request=0, **overrides)


def message(text):
    return {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def failed_body(output=None):
    return {**response_body(output or [], status="failed"),
            "error": {"code": "server_error", "message": "Service temporarily unavailable"}}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("dict_responses", [False, True])
@pytest.mark.parametrize("partial_output", [False, True])
def test_failed_response_retries_same_input_without_accepting_partial_output(
    monkeypatch, stream, dict_responses, partial_output
):
    requests = []
    executions = []
    failed_output = [
        {"id": "rs_bad", "type": "reasoning", "summary": [], "encrypted_content": "discard"},
        {"id": "fc_bad", "type": "function_call", "call_id": "bad", "name": "add",
         "arguments": '{"a":1,"b":2}', "status": "completed"},
        message("Incorrect partial output"),
    ] if partial_output else []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        result = failed_body(failed_output) if len(requests) == 1 else response_body([message("42")])
        return sse_response(result) if stream else httpx.Response(200, json=result)

    def add(a, b):
        executions.append((a, b))
        return str(a+b)

    install_transport(monkeypatch, handler, dict_responses)
    client = APIClient(**client_args(stream_openai_responses=stream, max_tool_calls=1, tools=[
        (add, {"type": "function", "function": {"name": "add", "parameters": {"type": "object"}}}),
    ]))
    [(idx, conversation, usage)] = client.run_queries([[{"role": "user", "content": "Question"}]], no_tqdm=True)
    assert idx == 0 and conversation[-1]["content"].strip() == "42"
    assert len(requests) == 2 and requests[0] == requests[1]
    assert executions == []
    assert "discard" not in json.dumps(conversation)
    assert usage["n_retries"] == 1 and not usage.get("request_failed")
    assert usage["output_tokens"] == 30


def test_failed_responses_exhaust_bounded_retries_and_are_marked_failed(monkeypatch):
    requests = []
    logged = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=failed_body())

    install_transport(monkeypatch, handler)
    monkeypatch.setattr(api_module.request_logger, "log_response", lambda **kw: logged.append(kw))
    [(idx, conversation, usage)] = APIClient(**client_args()).run_queries(
        [[{"role": "user", "content": "Question"}]], no_tqdm=True
    )
    assert len(requests) == 4  # Two outer attempts, two HTTP attempts each.
    assert all(r == requests[0] for r in requests)
    assert usage["request_failed"] is True
    assert conversation[-1]["content"] == ""
    assert len([x for x in logged if x.get("response", {}).get("status") == "failed"]) == 4
    errors = [x["exception"]["exception"] for x in logged if "exception" in x]
    assert len(errors) == 4 and all("Service temporarily unavailable" in e for e in errors)


@pytest.mark.parametrize("status,error", [("failed", None), ("cancelled", None),
                                          ("completed", {"code": "server_error", "message": "Failed"})])
def test_failure_status_or_error_is_rejected(monkeypatch, status, error):
    replies = [{**response_body([], status=status), "error": error}, response_body([message("42")])]
    install_transport(monkeypatch, lambda _: httpx.Response(200, json=replies.pop(0)))
    [(_, conversation, usage)] = APIClient(**client_args()).run_queries(
        [[{"role": "user", "content": "Question"}]], no_tqdm=True
    )
    assert conversation[-1]["content"].strip() == "42" and usage["n_retries"] == 1


def test_completed_empty_response_is_distinct_from_failed_request(monkeypatch):
    install_transport(monkeypatch, lambda _: httpx.Response(200, json=response_body([message("")])))
    [(_, conversation, usage)] = APIClient(**client_args()).run_queries(
        [[{"role": "user", "content": "Question"}]], no_tqdm=True
    )
    assert not conversation[-1]["content"].strip()
    assert not usage.get("request_failed") and usage["n_retries"] == 0


def test_failed_batch_item_is_marked_without_marking_success(monkeypatch):
    args = client_args(batch_processing=True, api_key_env="OPENROUTER_API_KEY")
    args.update(api="openai", use_openai_responses_api=False)
    client = APIClient(**args)
    assert client.batch_processing
    success = client.InternalRequestResult([{"role": "assistant", "content": "42"}], 10, 2)
    monkeypatch.setattr(client, "_openai_batch_processing", lambda *args: [None, success])
    results = list(client.run_queries([[{"role": "user", "content": str(i)}] for i in range(2)], no_tqdm=True))
    assert results[0][2]["request_failed"] is True
    assert not results[1][2].get("request_failed")


@pytest.mark.parametrize("details,expected", [
    ([{"type": "reasoning.summary", "summary": "Summary to preserve"},
      {"type": "reasoning.encrypted", "data": "opaque"}], "Summary to preserve"),
    ([{"type": "reasoning.text", "text": "Reasoning text"}], "Reasoning text"),
    ([{"type": "reasoning.encrypted", "data": "opaque"}], "Fallback reasoning"),
    ([{"type": "reasoning.summary", "summary": ""}], "Fallback reasoning"),
])
def test_chat_reasoning_details_and_fallback_are_preserved_once(monkeypatch, details, expected):
    body = {"id": "chat_1", "object": "chat.completion", "created": 0, "model": "x-ai/grok-4.7",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "42", "reasoning_details": details,
                "reasoning": "Fallback reasoning", "reasoning_content": "Fallback reasoning"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
    install_transport(monkeypatch, lambda _: httpx.Response(200, json=body))
    args = client_args();args["use_openai_responses_api"] = False
    [(_, conversation, _)] = APIClient(**args).run_queries(
        [[{"role": "user", "content": "Question"}]], no_tqdm=True
    )
    normalized = normalize_conversation(conversation)
    assert [m["content"] for m in normalized if m.get("type") == "cot"] == [expected]
    assert normalized[-1]["content"] == "42"


@pytest.mark.parametrize("fail_last_chance", [False, True])
def test_runner_leaves_api_failures_pending_and_saves_other_samples(tmp_path, monkeypatch, fail_last_chance):
    requests = []

    def handler(request):
        payload = json.loads(request.content);requests.append(payload)
        prompt = payload["input"][0]["content"]
        if prompt == "Success":return httpx.Response(200, json=response_body([message(r"\boxed{42}")]))
        if fail_last_chance and len(payload["input"]) == 1:
            return httpx.Response(200, json=response_body([message("Unparseable answer")]))
        return httpx.Response(200, json=failed_body())

    install_transport(monkeypatch, handler)
    args = client_args()
    solver = PureModelSolver({}, "{problem}", args, "Report the final answer")
    runner = Runner.__new__(Runner)
    runner.comp_name = "test";runner.runs_per_problem = 1
    runner.is_fa_comp = True;runner.is_auto_graded_comp = True;runner.options = None
    runner.competition_config = {"strict_parsing": True}
    runner.problems = [{"problem_idx": i, "problem": prompt, "answer": "42"}
                       for i, prompt in [(1, "Failure"), (2, "Success")]]
    all_runs = {problem["problem_idx"]: Runs("test", True, "grok", "pure_model", problem, str(tmp_path))
                for problem in runner.problems}
    monkeypatch.setattr(runner, "_update_status", lambda *args: None)
    responses = solver.solve_batch([("Failure", None), ("Success", None)], {0: 1, 1: 2}, {0: 0, 1: 0})
    runner.process_solver_responses("grok", solver, all_runs, {0: 1, 1: 2}, "unused", responses,
                                    print_final_status=False)
    assert all_runs[1].N == 0 and not Path(all_runs[1].path).exists()
    assert all_runs[2].N == 1 and load_json_zst(all_runs[2].path)["correct"] == [True]
    failed_requests = [r for r in requests if r["input"][0]["content"] == "Failure"]
    assert len(failed_requests) == (5 if fail_last_chance else 4)
    assert sum(len(r["input"]) > 1 for r in failed_requests) == (4 if fail_last_chance else 0)


def test_failed_background_response_is_checked_before_missing_usage(monkeypatch):
    methods = []
    logged = []

    def handler(request):
        methods.append(request.method)
        if len(methods) == 1:
            result = {**response_body([], status="queued"), "usage": None}
        elif request.method == "GET":
            result = {**failed_body(), "usage": None}
        else:
            result = response_body([message("42")])
        return httpx.Response(200, json=result)

    install_transport(monkeypatch, handler)
    monkeypatch.setattr(api_module.request_logger, "log_response", lambda **kw: logged.append(kw))
    [(_, conversation, usage)] = APIClient(**client_args(background=True)).run_queries(
        [[{"role": "user", "content": "Question"}]], no_tqdm=True
    )
    assert methods == ["POST", "GET", "POST"]
    assert conversation[-1]["content"].strip() == "42" and usage["n_retries"] == 1
    errors = [entry["exception"]["exception"] for entry in logged if "exception" in entry]
    assert len(errors) == 1 and "Service temporarily unavailable" in errors[0]
