"""Offline coverage of per-request long-context pricing and harness budgets."""

import json
from types import SimpleNamespace

import httpx
import pytest
from anthropic.types import Message
from harness_wrapper import AgentEvent, TokenUsage

import matharena.api_client as api_module
from matharena.api_client import APIClient
from matharena.solvers.harness_solver import HarnessSolver
from matharena.solvers.native_usage import NativeUsageReader
from matharena.solvers.run_budget import RunBudget
from matharena.token_pricing import TokenPricing
from test_meta_responses import install_transport, response_body, sse_response


PRICES = {
    "read_cost": 2, "cache_read_cost": .2, "cache_write_cost": 3, "write_cost": 10,
    "long_context_pricing": {
        "input_token_threshold": 100,
        "read_cost": 6, "cache_read_cost": .6, "cache_write_cost": 9, "write_cost": 15,
    },
}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(name, "test-only-key")
    monkeypatch.setattr(api_module.request_logger, "enabled", False)
    monkeypatch.setattr(api_module.time, "sleep", lambda _: None)
    def forbid_network(*args, **kwargs):
        raise AssertionError("Live network access is forbidden")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbid_network)


@pytest.mark.parametrize("input_tokens,output,cached,writes,expected", [
    (99, 500, 0, 0, 5198),  # Output cannot select the tier.
    (100, 20, 40, 0, 328),  # Equality retains ordinary rates.
    (101, 20, 80, 0, 474),  # Cache hits still count toward prompt length.
    (60, 20, 50, 40, 350),
    (60, 20, 50, 41, 759),  # Cache writes also select the tier.
])
def test_request_price_boundaries_and_all_token_categories(input_tokens, output, cached, writes, expected):
    pricing = TokenPricing.from_config(PRICES)
    assert pricing.cost(input_tokens, output, cached, writes) == pytest.approx(expected / 1e6)


@pytest.mark.parametrize("block", [
    {}, [], {"input_token_threshold": True},
    {"input_token_threshold": 0, "read_cost": 6, "write_cost": 15},
    {"input_token_threshold": 100, "read_cost": 6},
    {"input_token_threshold": 100, "read_cost": -1, "write_cost": 15},
    {"input_token_threshold": 100, "read_cost": 6, "write_cost": float("nan")},
    {"input_token_threshold": 100, "read_cost": 6, "write_cost": 15, "cache_read_cost": None},
    {"input_token_threshold": 100, "read_cost": 6, "write_cost": 15, "threshold": 100},
])
def test_invalid_pricing_fails_before_requests(block):
    with pytest.raises(ValueError, match="long_context_pricing"):
        APIClient(model="test", **(PRICES | {"long_context_pricing": block}))


def test_optional_cache_rates_keep_base_prices_and_zero_rates_are_allowed():
    pricing = TokenPricing.from_config(PRICES | {
        "long_context_pricing": {"input_token_threshold": 100, "read_cost": 0, "write_cost": 0},
    })
    assert pricing.cost(101, 100, 80, 1) == pytest.approx((80 * .2 + 3) / 1e6)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("inputs", [(80, 80), (100, 101), (120, 120)])
def test_responses_tool_loop_prices_each_call_and_does_not_send_accounting_config(monkeypatch, stream, inputs):
    replies = [
        response_body([{"id": "fc1", "type": "function_call", "call_id": "call1", "name": "add",
                        "arguments": '{"a":1,"b":2}', "status": "completed"}], inputs[0], 30, 20),
        response_body([{"id": "msg1", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "3", "annotations": []}]}], inputs[1], 50, 40),
    ]
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert "long_context_pricing" not in payload
        result = replies[len(requests) - 1]
        return sse_response(result) if stream else httpx.Response(200, json=result)
    install_transport(monkeypatch, handler)
    client = APIClient(model="test", use_openai_responses_api=True, stream_openai_responses=stream,
                       max_retries=1, max_retries_inner=0, max_tool_calls=1, tools=[
                           (lambda a, b: str(a + b), {"type": "function", "function": {
                               "name": "add", "parameters": {"type": "object"},
                           }}),
                       ], **PRICES)
    [(_, _, usage)] = client.run_queries([[{"role": "user", "content": "Add"}]], no_tqdm=True)
    pricing = TokenPricing.from_config(PRICES)
    expected = pricing.cost(inputs[0], 30, 20) + pricing.cost(inputs[1], 50, 40)
    assert len(requests) == 2
    assert usage["input_tokens"] == sum(inputs)
    assert usage["cost"] == pytest.approx(expected)


def test_chat_tool_loop_does_not_select_tier_from_accumulated_input(monkeypatch):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call1", "type": "function", "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
        }]} if len(requests) == 1 else {"role": "assistant", "content": "3"}
        return httpx.Response(200, json={
            "id": f"chat{len(requests)}", "object": "chat.completion", "created": 0, "model": "test",
            "choices": [{"index": 0, "finish_reason": "tool_calls" if len(requests) == 1 else "stop", "message": message}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
        })
    install_transport(monkeypatch, handler)
    client = APIClient(model="test", max_retries=1, max_retries_inner=0, max_tool_calls=2,
                       tools=[(lambda a, b: str(a+b), {"type": "function", "function": {
                           "name": "add", "parameters": {"type": "object"},
                       }})], **PRICES)
    [(_, _, usage)] = client.run_queries([[{"role": "user", "content": "Add"}]], no_tqdm=True)
    assert len(requests) == 2
    assert usage["cost"] == pytest.approx(2 * (80*2 + 20*10) / 1e6)


@pytest.mark.parametrize("writes,expected", [(40, 350), (41, 759)])
def test_anthropic_prompt_length_includes_cache_reads_and_writes(monkeypatch, writes, expected):
    requests = []
    def create(**payload):
        requests.append(payload)
        return Message.model_validate({
            "id": "msg1", "type": "message", "role": "assistant", "model": "test",
            "stop_reason": "end_turn", "stop_sequence": None,
            "content": [{"type": "text", "text": "42"}],
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 50,
                      "cache_creation_input_tokens": writes, "output_tokens": 20},
        })
    monkeypatch.setattr(api_module.anthropic, "Anthropic", lambda **kw: SimpleNamespace(messages=SimpleNamespace(create=create)))
    client = APIClient(model="test", api="anthropic", max_tokens=100, **PRICES)
    [(_, _, usage)] = client.run_queries([[{"role": "user", "content": "Question"}]], no_tqdm=True)
    assert "long_context_pricing" not in requests[0]
    assert usage["cost"] == pytest.approx(expected / 1e6)


@pytest.mark.parametrize("input_tokens,writes,expected", [(100, 40, 404), (101, 41, 921)])
def test_openai_cache_writes_are_a_subset_of_input_in_api_and_native_usage(monkeypatch, tmp_path, input_tokens, writes, expected):
    body = response_body([{"id": "msg", "type": "message", "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": "42", "annotations": []}]}],
                         input_tokens, 20, 20)
    body["usage"]["input_tokens_details"]["cache_write_tokens"] = writes
    install_transport(monkeypatch, lambda _: httpx.Response(200, json=body))
    client = APIClient(model="test", use_openai_responses_api=True, **PRICES)
    [(_, _, usage)] = client.run_queries([[{"role": "user", "content": "Q"}]], no_tqdm=True)
    assert usage["cost"] == pytest.approx(expected / 1e6)
    assert usage["cached_write_tokens"] == writes

    reader = NativeUsageReader(tmp_path, "codex")
    path = tmp_path / ".harness-home/.codex/sessions/session.jsonl"
    path.parent.mkdir(parents=True)
    row = codex_record(input_tokens + 20, input_tokens)
    row["payload"]["info"]["last_token_usage"].update(cache_write_input_tokens=writes, output_tokens=20,
                                                    total_tokens=input_tokens + 20)
    path.write_text(json.dumps(row) + "\n")
    s = solver(tmp_path)
    b = request_budget(s, reader=reader)
    native = TokenUsage(input_tokens=input_tokens, cache_read_tokens=20, output_tokens=20)
    assert b.cost(native) == pytest.approx(expected / 1e6)
    assert b.usage().cache_write_tokens == writes
    assert not b.cost_estimated
    assert s._detailed_cost(None, usage=b.usage(), budget=b)["cost"] == pytest.approx(expected / 1e6)


@pytest.mark.parametrize("api", ["openai", "anthropic", "openrouter"])
def test_batch_prices_individual_items_and_preserves_existing_discount(monkeypatch, api):
    client = APIClient(model="anthropic/test" if api == "openrouter" else "test",
                       api=api, batch_processing=True, **PRICES)
    results = [client.InternalRequestResult([], n, 20) for n in (100, 101)]
    monkeypatch.setattr(client, f"_{api}_batch_processing", lambda *args: results)
    usages = list(client.run_queries([[{"role": "user", "content": "Q"}]]*2, no_tqdm=True))
    factor = .5 if api == "openrouter" else 1
    assert usages[0][2]["cost"] == pytest.approx(400 * factor / 1e6)
    assert usages[1][2]["cost"] == pytest.approx(906 * factor / 1e6)


def test_streamed_chat_prices_final_usage_with_cache_writes_once(monkeypatch):
    def handler(request):
        assert "long_context_pricing" not in json.loads(request.content)
        chunks = [
            {"id": "r1", "object": "chat.completion.chunk", "created": 0, "model": "test",
             "choices": [{"index": 0, "delta": {"content": "42"}, "finish_reason": "stop"}]},
            {"id": "r1", "object": "chat.completion.chunk", "created": 0, "model": "test", "choices": [],
             "usage": {"prompt_tokens": 101, "completion_tokens": 20, "total_tokens": 121,
                       "prompt_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 41}}},
        ]
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                              content="".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n")
    install_transport(monkeypatch, handler)
    client = APIClient(model="test", stream_openai_chat_completions=True, **PRICES)
    [(_, conversation, usage)] = client.run_queries([[{"role": "user", "content": "Q"}]], no_tqdm=True)
    assert conversation[-1]["content"] == "42"
    assert usage["cost"] == pytest.approx(.000921)
    assert usage["cached_write_tokens"] == 41


def solver(tmp_path, harness="codex"):
    config = PRICES | {"model": "test", "api": "anthropic" if harness == "claude" else "openai", "harness": harness,
                       "harness_config": {"managed_cli": False, "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}")}}
    return HarnessSolver({}, "{problem}", config, "Answer now")


def request_budget(solver, limits=None, reader=None):
    return RunBudget(limits or {}, lambda u: solver._detailed_cost(None, usage=u)["cost"], priced=True,
                     request_level_cost=True, native_usage_reader=reader,
                     base_cost_for_usage=lambda u: solver._detailed_cost(None, usage=u, base_only=True)["cost"])


def test_budget_uses_per_request_prices_and_deduplicates_streaming(tmp_path):
    s = solver(tmp_path, "claude")
    b = request_budget(s, {"max_cost_usd": .0006})
    for response_id in ("r1", "r2"):
        b.capture_response("/v1/messages", {"type": "message_start", "message": {
            "id": response_id, "usage": {"input_tokens": 60, "output_tokens": 0},
        }})
        for output in (5, 10, 10):
            b.capture_response("/v1/messages", {"type": "message_delta", "usage": {"output_tokens": output}})
    assert b.cost() == pytest.approx(.00044)
    assert not b.exceeded()
    assert s._detailed_cost(None, usage=b.usage(), budget=b)["cost"] == pytest.approx(.00044)
    b.capture_response("/v1/messages", {"id": "r3", "usage": {"input_tokens": 101, "output_tokens": 10}})
    assert b.exceeded() and b.reason == "cost_limit"
    assert b.cost() == pytest.approx(.001196)


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_harness_captures_requests_without_limits_and_keeps_cost_on_resume(tmp_path, monkeypatch, harness):
    s = solver(tmp_path, harness)
    class FakeAgent:
        session_id = "session"
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.usage = TokenUsage()
        def get_tokens(self):
            return self.usage
        def record(self, response_id, input_tokens):
            payload = {"id": response_id, "usage": {"input_tokens": input_tokens, "output_tokens": 10}}
            for _ in range(2):
                self.kwargs["on_model_response"]("/v1/responses", payload)
            self.usage = TokenUsage(input_tokens=self.usage.input_tokens + input_tokens,
                                    output_tokens=self.usage.output_tokens + 10)
        def run(self, prompt):
            assert self.kwargs["should_stop"] is None
            self.record("r1", 120)
            self.record("r2", 60)
            return [AgentEvent(type="result", content="42")]
        def resume(self, prompt, session_id):
            self.record("r3", 70)
            return [AgentEvent(type="result", content="42")]
    monkeypatch.setattr("matharena.solvers.harness_solver.Agent", FakeAgent)
    monkeypatch.setattr(s, "_prepare_harness_cli", lambda: None)
    monkeypatch.setattr(s, "_build_model", lambda: None)
    monkeypatch.setattr(s, "_build_sandbox", lambda workspace: SimpleNamespace())
    response = s._solve_one(0, "Q", None, 1, 0)
    assert response.detailed_cost["cost"] == pytest.approx(.00109)
    assert "run_limits" not in response.detailed_cost
    assert not response.detailed_cost.get("long_context_cost_estimated")
    response = s.last_chance(response)
    assert response.detailed_cost["cost"] == pytest.approx(.00133)
    assert response.detailed_cost["input_tokens"] == 250
    assert "run_limits" not in response.detailed_cost


def codex_record(total, input_tokens):
    return {"type": "event_msg", "payload": {"type": "token_count", "info": {
        "total_token_usage": {"total_tokens": total},
        "last_token_usage": {"input_tokens": input_tokens, "cached_input_tokens": 20,
                             "output_tokens": 10, "total_tokens": input_tokens + 10},
    }}}


def test_native_codex_records_are_incremental_and_price_each_request(tmp_path):
    reader = NativeUsageReader(tmp_path, "codex")
    path = tmp_path / ".harness-home/.codex/sessions/2026/10/12/session.jsonl"
    path.parent.mkdir(parents=True)
    records = [codex_record(130, 120)]*2 + [codex_record(200, 60)]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    b = request_budget(solver(tmp_path), reader=reader)
    native = TokenUsage(input_tokens=180, cache_read_tokens=40, output_tokens=20)
    # High request: 100*6 + 20*.6 + 10*15; low: 40*2 + 20*.2 + 10*10.
    assert b.cost(native) == pytest.approx(.000946)
    assert b.cost(native) == pytest.approx(.000946)
    assert not b.cost_estimated
    final = json.dumps(codex_record(270, 60))
    with path.open("a") as f:
        f.write(final[:20])
    assert len(reader.usages()) == 2
    with path.open("a") as f:
        f.write(final[20:] + "\n")
    assert len(reader.usages()) == 3  # Identical usage from a distinct request counts again.


def test_native_claude_deduplicates_message_ids_and_includes_caches(tmp_path):
    reader = NativeUsageReader(tmp_path, "claude")
    path = tmp_path / ".harness-home/.claude/projects/work/session.jsonl"
    path.parent.mkdir(parents=True)
    def record(message_id, output):
        return {"type": "assistant", "message": {"id": message_id, "usage": {
            "input_tokens": 10, "cache_read_input_tokens": 50, "cache_creation_input_tokens": 41,
            "output_tokens": output,
        }}}
    path.write_text("\n".join(json.dumps(r) for r in [record("r1", 5), record("r1", 20), record("r2", 20)]) + "\n")
    b = request_budget(solver(tmp_path, "claude"), reader=reader)
    assert b.cost(TokenUsage(input_tokens=20, cache_read_tokens=100, cache_write_tokens=82, output_tokens=40)) == pytest.approx(.001518)
    assert not b.cost_estimated


def test_native_reader_ignores_old_attempt_and_marks_unattributed_usage(tmp_path):
    path = tmp_path / ".harness-home/.codex/sessions/old.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(codex_record(130, 120)) + "\n")
    reader = NativeUsageReader(tmp_path, "codex")
    assert reader.usages() == []
    s = solver(tmp_path)
    b = request_budget(s, reader=reader)
    native = TokenUsage(input_tokens=200, output_tokens=20)
    cost = s._detailed_cost(None, usage=native, budget=b)
    assert cost["cost"] == pytest.approx(.0006)  # Unknown request lengths use ordinary rates.
    assert cost["long_context_cost_estimated"]
