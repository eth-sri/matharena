import pytest

from leanlean.result_trace import join_fields, split_file
from matharena.json_zst import dump_json_zst, load_json_zst


def test_result_and_trace_reconstruct_all_original_fields(tmp_path):
    data = dict(idx=1, cost={"cost": 2.5}, messages=[dict(role="assistant", content="λ")],
                history=[[dict(events=[dict(type="tool", output="exact")])]], correct=False,
                warnings=[], detailed_costs=[dict(output_tokens=10)])
    path = tmp_path / "result.json.zst"
    dump_json_zst(data, path)
    receipt = split_file(path)
    result, trace = load_json_zst(path), load_json_zst(tmp_path / "agent-trace.json.zst")
    assert "messages" not in result and "history" not in result
    assert result["cost"] == {"cost": 2.5}
    assert join_fields(result, trace) == data
    assert join_fields({**result, "leanlean": {"compression_percent": 42}}, trace) == data
    assert receipt["original_json_bytes_reconstructable"]
    with pytest.raises(ValueError, match="overlap"):
        join_fields({**result, "history": []}, trace)


def test_no_history_still_preserves_the_result(tmp_path):
    path = tmp_path / "result.json.zst"
    data = dict(idx=1, answers=["final answer"], correct=True)
    dump_json_zst(data, path)
    split_file(path)
    assert join_fields(load_json_zst(path), load_json_zst(tmp_path / "agent-trace.json.zst")) == data
