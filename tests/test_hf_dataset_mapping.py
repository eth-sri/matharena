import pytest

import matharena.runner as runner_module
from matharena.runner import Runner


class FakeDataset:
    def __init__(self, rows):
        self._rows = rows

    def to_list(self):
        return [dict(row) for row in self._rows]


def load_mapped_problems(monkeypatch, config, rows):
    runner = Runner.__new__(Runner)
    runner.competition_config = config
    runner.is_lean_comp = False
    runner.is_constructive_comp = False
    monkeypatch.setattr(runner_module.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(runner_module, "load_dataset", lambda _path, split: FakeDataset(rows))
    return runner._load_problems(None)


def test_hf_dataset_problem_column_and_constant_answer(monkeypatch):
    problems = load_mapped_problems(
        monkeypatch,
        {
            "dataset_path": "MathArena/brokenarxiv-training",
            "split": "train",
            "problem_column": "perturbed_statement",
            "answer_value": "False",
        },
        [
            {
                "paper_id": "1234.56789",
                "perturbed_statement": "This theorem is false.",
            }
        ],
    )

    assert problems == [
        {
            "paper_id": "1234.56789",
            "perturbed_statement": "This theorem is false.",
            "problem_idx": 1,
            "problem": "This theorem is false.",
            "answer": "False",
            "statement": "This theorem is false.",
            "image": None,
            "source": "1234.56789",
        }
    ]


def test_hf_dataset_answer_column_conflicts_with_answer_value(monkeypatch):
    with pytest.raises(ValueError, match="answer_column and answer_value"):
        load_mapped_problems(
            monkeypatch,
            {
                "dataset_path": "example/dataset",
                "answer_column": "label",
                "answer_value": "False",
            },
            [],
        )
