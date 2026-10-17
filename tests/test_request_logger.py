import importlib.util
import json
from pathlib import Path

from matharena.request_logger import RequestLogger


ROOT = Path(__file__).resolve().parents[1]


def _load_run_module():
    spec = importlib.util.spec_from_file_location("matharena_run", ROOT / "scripts" / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_competition_request_log_keeps_comp_name(tmp_path):
    request_logger = RequestLogger()
    request_logger.log_dir = str(tmp_path)
    request_logger.enabled = True
    request_logger.set_metadata("aime/aime_2026", "model", {0: 1})

    request_logger.log_request("single", 0, {"prompt": "test"})

    log_path = tmp_path / "aime/aime_2026" / "model" / "single_p1_idx0.json"
    data = json.loads(log_path.read_text(encoding="utf-8"))
    assert data["comp_name"] == "aime/aime_2026"
    assert data["problem_idx"] == 1


def test_multi_competition_request_log_preserves_source_comp_name(tmp_path):
    request_logger = RequestLogger()
    request_logger.log_dir = str(tmp_path)
    request_logger.enabled = True
    request_logger.set_metadata(
        "multi",
        "model",
        {0: 1, 1: 1},
        {
            0: "aime/aime_2026",
            1: "hmmt/hmmt_feb_2026",
        },
    )

    request_logger.log_request("aime", 0, {"prompt": "aime"})
    request_logger.log_request("hmmt", 1, {"prompt": "hmmt"})

    log_dir = tmp_path / "multi" / "model"
    aime = json.loads((log_dir / "aime_p1_idx0.json").read_text(encoding="utf-8"))
    hmmt = json.loads((log_dir / "hmmt_p1_idx1.json").read_text(encoding="utf-8"))

    assert aime["comp_name"] == "aime/aime_2026"
    assert hmmt["comp_name"] == "hmmt/hmmt_feb_2026"
    assert aime["problem_idx"] == hmmt["problem_idx"] == 1


def test_combined_group_passes_per_batch_competition_metadata(monkeypatch):
    run = _load_run_module()
    captured = {}

    def capture_metadata(*args):
        captured["args"] = args

    monkeypatch.setattr(run.request_logger, "set_metadata", capture_metadata)

    class FakeClient:
        def run_queries(self, queries):
            return []

    class FakeSolver:
        client = FakeClient()

        def build_query(self, text, image):
            return [text, image]

    class FakeRunner:
        def __init__(self, comp_name):
            self.comp_name = comp_name

        def print_final_status(self, _status_path):
            pass

    group = [
        (
            FakeRunner("aime/aime_2026"),
            {
                "solver": FakeSolver(),
                "batch": [("aime", None)],
                "batch_idx_to_problem_idx": {0: 1},
                "status_path": "aime-status",
            },
        ),
        (
            FakeRunner("hmmt/hmmt_feb_2026"),
            {
                "solver": FakeSolver(),
                "batch": [("hmmt", None)],
                "batch_idx_to_problem_idx": {0: 1},
                "status_path": "hmmt-status",
            },
        ),
    ]

    run._run_combined_pure_model_group("model", group)

    assert captured["args"] == (
        "multi",
        "model",
        {0: 1, 1: 1},
        {0: "aime/aime_2026", 1: "hmmt/hmmt_feb_2026"},
    )
