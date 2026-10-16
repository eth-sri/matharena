"""Public competition configs must work without the internal data checkout."""

from pathlib import Path

import pytest
import yaml

from matharena.runner import Runner

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs/competitions").glob("arxiv*/*.yaml"))
MONTHS = {
    "december": "1225", "january": "0126", "february": "0226",
    "march": "0326", "april": "0426", "may": "0526", "june": "0626",
    "august": "0826", "september": "0926",
}
PREFIXES = {"arxiv": "arxivmath", "arxiv_false": "brokenarxiv", "arxivlean": "arxivlean"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_arxiv_competitions_use_the_published_dataset_name(path):
    config = yaml.safe_load(path.read_text())
    prefix = PREFIXES[path.parent.name]
    month = path.stem.split("_")[0]
    expected = f"MathArena/{prefix}"
    if month != "all":
        expected += "-" + MONTHS[month]
    assert config["dataset_path"] == expected
    assert config["n_problems"] > 0


@pytest.mark.parametrize("competition", ["arxiv", "arxiv_false", "arxivlean"])
def test_september_hf_rows_load_without_local_dataset_files(monkeypatch, competition):
    runner = Runner.__new__(Runner)
    runner.competition_config = yaml.safe_load(
        (ROOT / f"configs/competitions/{competition}/september.yaml").read_text()
    )
    runner.is_lean_comp = runner.competition_config.get("lean", False)
    row = {"problem_idx": 1, "problem": "A test problem."}
    if runner.is_lean_comp:
        row.update(answer="def arxivleanConjecture : Prop := True",
                   formal_statement="def arxivleanConjecture : Prop := True")
    elif competition == "arxiv":
        row["answer"] = "1"
    else:
        row.update(points=3, original_problem="The original statement.")

    class Dataset:
        def to_list(self):
            return [dict(row)]

    calls = []

    def load_dataset(name, split):
        calls.append((name, split))
        return Dataset()

    monkeypatch.setattr("matharena.runner.load_dataset", load_dataset)
    monkeypatch.setattr("matharena.runner.os.path.exists", lambda _: False)
    [problem] = runner._load_problems([1])
    assert calls == [(runner.competition_config["dataset_path"], "train")]
    assert problem["problem_idx"] == 1
    assert problem["statement"] == row["problem"]
    assert problem["image"] is None
    if runner.is_lean_comp:
        assert problem["formal_statement"] == problem["answer"]
