"""Malformed saved attempts fail before any judge request or output write."""

import importlib.util
from pathlib import Path

import pytest

from matharena.json_zst import dump_json_zst


@pytest.mark.parametrize(
    "conversation",
    [
        [],
        [{"role": "assistant", "type": "tool_call", "arguments": {}}],
        [{"role": "tool_response", "content": "Not an answer"}],
        [{"role": "assistant", "type": "cot", "content": "Still thinking"}],
        [{"role": "assistant", "type": "response"}],
    ],
)
@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_judge_rejects_incomplete_transcript_before_requests(
    tmp_path, monkeypatch, conversation, runtime
):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "judge_preflight_test", root / "scripts/judge/judge.py"
    )
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    path = tmp_path / "arxiv/august/gemini/gemini-38-flash/4.json.zst"
    path.parent.mkdir(parents=True)
    dump_json_zst(
        {
            "idx": 4,
            "problem": "Question",
            "messages": [conversation],
            "answers": ["None"],
        },
        path,
    )
    before = path.read_bytes()

    class NoRequests:
        def __init__(self, **kwargs):
            assert kwargs["container_runtime"] == runtime

        def solve_batch(self, *args):
            pytest.fail("Incomplete attempts must be rejected before any grading calls")

    monkeypatch.setattr(judge, "JudgePool", NoRequests)
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        "sys.argv",
        [
            "judge.py",
            "--comp",
            "arxiv/august",
            "--output-dir",
            str(tmp_path),
            "--models",
            "gemini/gemini-38-flash",
            "--container-runtime",
            runtime,
        ],
    )
    with pytest.raises(ValueError, match="run 1: no final assistant response"):
        judge.main()
    assert path.read_bytes() == before
