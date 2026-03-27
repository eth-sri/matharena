import json
import os
from collections import OrderedDict
import time

from loguru import logger

from matharena.json_zst import OUTPUT_JSON_SUFFIX, dump_json_zst, load_json_zst
from matharena.utils import (
    convert_answer_to_string,
    lists_differ,
    is_conversation_broken,
    normalize_conversation,
    save_run_for_recovery,
)


class Runs:
    """
    List of all runs of a particular solver for a specific problem in a specific comp.
    Corresponds to outputs/{comp}/{solver}/{problem_idx}.json.zst files.
    """

    def __init__(self, comp_name, is_fa, solver_name, solver_type, problem, output_dir):
        self.sorted_keys = [
            # Fixed problem info:
            "idx",
            "problem",
            "gold_answer",
            "source",
            "types",
            # Aggregates:
            "N",
            "cost",
            "pass_at_1",
            # Results of grading:
            "answers",
            "correct",
            "warnings",
            # Raw data:
            "messages",
            "judgment",
            "history",
            "detailed_costs",
        ]
        self.optional_keys = [
            "manual_overwrite",
            "llm_annotation",
        ]
        self.is_fa = is_fa
        self.comp_name = comp_name
        self.solver_name = solver_name
        self.solver_type = solver_type
        self.output_dir = output_dir

        # Save problem info
        self.problem_idx = int(problem["problem_idx"])
        self.problem_stmt = problem["problem"] if "problem" in problem else None  # NOTE: we always drop the image to save space
        self.gold_answer = str(problem["answer"]) if is_fa else None  # no final answer
        self.source = problem.get("source", "None")
        self.problem_type = problem.get("problem_type", [])
        # NOTE: singular name for consistency with HF but is a list

        # Initialize to default
        self.N, self.cost, self.pass_at_1 = 0, {}, None
        self.answers, self.correct, self.warnings = [], [], []
        self.manual_overwrite, self.llm_annotation = [], []
        self.messages, self.judgment, self.history, self.detailed_costs = [], [], [], []

        # Path
        self.path = f"{output_dir}/{self.problem_idx}{OUTPUT_JSON_SUFFIX}"
        self.message_keys_zorder = {
            "role": 0,
            "type": 1,
            "tool_name": 2,
            "tool_call_id": 3,
            "arguments": 4,
            "content": 100,
        }
        self.history_keys_zorder = {
            "step": 0,
            "timestep": 1,
            "messages": 2,
        }

    def load_from_file(self):
        """Loads runs from a JSON file."""
        if not os.path.exists(self.path):
            return
        runs_dict = load_json_zst(self.path)
        self.from_dict(runs_dict)

    def save_to_file(self):
        """Saves runs to a JSON file."""
        runs_dict = self.to_dict()
        if self.N == 0:
            return
        dump_json_zst(runs_dict, self.path, indent=4, ensure_ascii=False)

    def from_dict(self, runs_dict):
        """Loads the object from a dictionary.
        Args:
            runs_dict (dict): The dictionary representation of problem runs.
        Returns:
            None
        """
        # Report missing and extra keys
        for k in self.sorted_keys:
            if k not in runs_dict:
                logger.warning(f"Missing key {k} in {self.path}")
        for k in runs_dict:
            if k not in self.sorted_keys and k not in self.optional_keys:
                logger.warning(f"Extra key {k} in {self.path}")

        # Check problem info is the same
        if self.problem_idx != runs_dict.get("idx"):
            if self.problem_idx == runs_dict.get("idx") + 1:
                # Previously it was off by one; now we are fixing it but idx field is unused anyways
                logger.warning(
                    f"Problem index off by one in {self.path}: here {self.problem_idx} vs there {runs_dict.get('idx')}"
                )
            else:
                logger.warning(
                    f"Problem index mismatch in {self.path}: here {self.problem_idx} vs there {runs_dict.get('idx')}"
                )
        if self.problem_stmt != runs_dict.get("problem"):
            logger.warning(
                f"Problem statement mismatch in {self.path}: here {self.problem_stmt} vs there {runs_dict.get('problem')}"
            )
        if self.gold_answer != runs_dict.get("gold_answer"):
            logger.warning(
                f"Gold answer mismatch in {self.path}: here {self.gold_answer} vs there {runs_dict.get('gold_answer')}"
            )
        if self.source != runs_dict.get("source"):
            logger.warning(f"Source mismatch in {self.path}: here {self.source} vs there {runs_dict.get('source')}")
        if "types" not in runs_dict:
            logger.warning(f"Missing problem types in {self.path}")
        elif lists_differ(self.problem_type, runs_dict["types"]):
            logger.warning(
                f"Problem type mismatch in {self.path}: here {self.problem_type} vs there {runs_dict['types']}"
            )

        # Load aggregates
        self.N = len(runs_dict["answers"])
        self.cost = runs_dict["cost"]
        self.pass_at_1 = runs_dict["pass_at_1"]

        # Load grading results
        self.answers = runs_dict["answers"]
        assert len(self.answers) == self.N
        self.correct = runs_dict["correct"]
        assert len(self.correct) == self.N
        self.warnings = runs_dict["warnings"]
        assert len(self.warnings) == self.N
        self.manual_overwrite = runs_dict.get("manual_overwrite", [False for _ in range(self.N)])
        self.llm_annotation = runs_dict.get("llm_annotation", [None for _ in range(self.N)])

        # Load raw data, if missing patch
        self.messages = runs_dict["messages"]
        assert len(self.messages) == self.N
        self.judgment = runs_dict.get("judgment", [None for _ in range(self.N)])
        self.history = runs_dict.get("history", [[None] for _ in range(self.N)])
        assert len(self.history) == self.N
        for i, h in enumerate(self.history):
            if isinstance(h, list) and h[0] is None:  # TMP
                self.history[i] = None
            self.validate_history(self.history[i])
        if "detailed_costs" in runs_dict:
            self.detailed_costs = runs_dict["detailed_costs"]
        else:
            # Just put all cost in first run if we never had details
            self.detailed_costs = [self.cost.copy()]
            self.detailed_costs.extend([{"cost": 0, "input_tokens": 0, "output_tokens": 0} for _ in range(self.N - 1)])
        for i in range(self.N):
            if "time" not in self.detailed_costs[i]:
                self.detailed_costs[i]["time"] = None
        assert len(self.detailed_costs) == self.N

        # Check total cost
        for k in ["cost", "input_tokens", "output_tokens", "time"]:
            if k not in self.cost:
                logger.warning(f"Missing total {k} in {self.path}")
                if any(dc.get(k, None) is None for dc in self.detailed_costs):
                    self.cost[k] = None
                else:
                    self.cost[k] = sum(dc[k] for dc in self.detailed_costs)
            vals = [dc[k] for dc in self.detailed_costs]
            if any([v is None for v in vals]):
                if self.cost[k] is not None:
                    logger.warning(f"Total {k} mismatch in {self.path}: total {self.cost[k]} vs some details None")
            else:
                if self.cost[k] != sum(vals):
                    logger.warning(
                        f"Total {k} mismatch in {self.path}: aggregate {self.cost[k]} vs sum of details {sum(dc[k] for dc in self.detailed_costs)}"
                    )

        # Check for shady messages that should have been retries and drop them
        indices_to_drop = []
        for i in range(self.N):
            is_broken, reason = is_conversation_broken(self.messages[i])
            if is_broken:
                logger.warning(f"Message list {i} is broken: {reason}, DROPPING")
                indices_to_drop.append(i)
        if len(indices_to_drop) > 0:
            self.drop_runs(indices_to_drop)

    def to_dict(self):
        """Converts the object to an ordered dictionary.

        Returns:
            dict: The ordered dictionary representation of problem runs.
        """
        ordered_messages = []
        for i in range(self.N):
            convo = self.messages[i]
            ordered_convo = []
            for block in convo:
                block_kv = sorted(block.items(), key=lambda x: self.message_keys_zorder.get(x[0], 10))
                ordered_convo.append(OrderedDict(block_kv))
            ordered_messages.append(ordered_convo)

        ordered_history = []
        for i in range(self.N):
            h = self.history[i]
            if h is None:
                ordered_history.append(None)
                continue
            ordered_h = []
            for step in h:
                ordered_step = OrderedDict(sorted(step.items(), key=lambda x: self.history_keys_zorder.get(x[0], 10)))
                if "messages" not in step:
                    import code

                    code.interact(local=dict(globals(), **locals()))
                convo = step["messages"]
                ordered_convo = []
                for block in convo:
                    block_kv = sorted(block.items(), key=lambda x: self.message_keys_zorder.get(x[0], 10))
                    ordered_convo.append(OrderedDict(block_kv))
                ordered_step["messages"] = ordered_convo
                ordered_h.append(ordered_step)
            ordered_history.append(ordered_h)

        return OrderedDict(
            [
                ("idx", self.problem_idx),
                ("problem", self.problem_stmt),
                ("gold_answer", self.gold_answer),
                ("source", self.source),
                ("types", self.problem_type),
                ("N", self.N),
                ("cost", self.cost),
                ("pass_at_1", self.pass_at_1),
                ("answers", self.answers),
                ("correct", self.correct),
                ("warnings", self.warnings),
                ("manual_overwrite", self.manual_overwrite),
                ("llm_annotation", self.llm_annotation),
                ("messages", ordered_messages),
                ("judgment", self.judgment),
                ("history", ordered_history),
                ("detailed_costs", self.detailed_costs),
            ]
        )

    def update_aggregates(self):
        self.N = len(self.messages)
        self.cost = {}
        for k in ["cost", "input_tokens", "output_tokens", "time", "n_retries", "request_time"]:
            vals = [dc.get(k, None) for dc in self.detailed_costs]
            if any(v is None for v in vals):
                self.cost[k] = None
            else:
                self.cost[k] = sum(vals)
        if not self.is_fa:
            self.pass_at_1 = None
        if self.N == 0:
            self.pass_at_1 = None
        elif any(isinstance(c, str) for c in self.correct):
            self.pass_at_1 = "TODO Grading"
        else:
            self.pass_at_1 = sum(self.correct) / self.N

    def add_run(self, solver_response, grader_response):
        """
        Grades a run from a SolverResponse object and a GraderResponse object and adds to runs.
        Normalizes both conversation and history before adding.

        Args:
            solver_response (SolverResponse): The response from the solver for this problem.
            grader_response (tuple): The response from the grader as (answer, is_correct, warning).
        """
        # Solver response: idx, messages, detailed_cost, history
        # Aggregates: N, cost, pass_at_1
        # Grader Response: answers, correct, warnings
        # Raw: messages, judgement, history, detailed_costs

        # Clean and validate messages and history
        try:
            clean_conversation = normalize_conversation(solver_response.conversation)
            history = solver_response.history
            self.validate_history(history)
            if history is not None:
                for i in range(len(history)):
                    history[i]["messages"] = normalize_conversation(history[i]["messages"])
        except Exception as e:  # noqa E722
            logger.error(f"Error during normalization or history validation: {e}")
            save_run_for_recovery("runs add_run", self.path, solver_response, grader_response)
            raise

        self.messages.append(clean_conversation)
        self.judgment.append(None)
        self.history.append(history)
        self.detailed_costs.append(solver_response.detailed_cost)
        answer, is_correct, warning = grader_response
        self.answers.append(convert_answer_to_string(answer) if answer is not None else "None")
        self.correct.append(is_correct)
        self.warnings.append(warning)
        self.manual_overwrite.append(False)
        self.llm_annotation.append(None)

        self.update_aggregates()

    def update_run_grading(self, idx, grader_response):
        answer, is_correct, warning = grader_response
        self.answers[idx] = convert_answer_to_string(answer) if answer is not None else "None"
        if self.manual_overwrite and idx < len(self.manual_overwrite) and self.manual_overwrite[idx]:
            self.warnings[idx] = warning
        else:
            self.correct[idx] = is_correct
            self.warnings[idx] = warning

        self.update_aggregates()

    def update_run_costs(self, idx, new_in_cost, new_out_cost, new_cache_in_cost=None, new_cache_write_cost=None):
        in_toks = self.detailed_costs[idx]["input_tokens"]
        out_toks = self.detailed_costs[idx]["output_tokens"]
        cached_in_toks = self.detailed_costs[idx].get("cached_input_tokens", 0)
        cached_in_toks = min(max(cached_in_toks, 0), max(in_toks, 0))
        cached_write_tokens = self.detailed_costs[idx].get("cached_write_tokens", 0)
        uncached_in_toks = in_toks - cached_in_toks
        if new_cache_in_cost is None:
            new_cache_in_cost = new_in_cost
        if new_cache_write_cost is None:
            new_cache_write_cost = new_out_cost
        new_cost = (uncached_in_toks * new_in_cost + cached_in_toks * new_cache_in_cost + out_toks * new_out_cost + cached_write_tokens * new_cache_write_cost) / 1e6
        self.detailed_costs[idx]["cost"] = new_cost

        self.update_aggregates()

    def drop_runs(self, indices_to_drop):
        indices_kept = [i for i in range(self.N) if i not in indices_to_drop]
        self.answers = [self.answers[i] for i in indices_kept]
        self.correct = [self.correct[i] for i in indices_kept]
        self.warnings = [self.warnings[i] for i in indices_kept]
        self.manual_overwrite = [self.manual_overwrite[i] for i in indices_kept]
        self.llm_annotation = [self.llm_annotation[i] for i in indices_kept]

        self.messages = [self.messages[i] for i in indices_kept]
        self.judgment = [self.judgment[i] for i in indices_kept]
        self.history = [self.history[i] for i in indices_kept]
        self.detailed_costs = [self.detailed_costs[i] for i in indices_kept]

        self.update_aggregates()

    def validate_history(self, history):
        if self.solver_type == "pure_model":
            assert history is None, f"History should be None for pure_model but got: {history}"
            return

        # Agents
        assert isinstance(history, list)
        steps = set()
        for h in history:
            assert isinstance(h, dict)
            assert "step" in h and isinstance(h["step"], str)
            assert "timestep" in h and isinstance(h["timestep"], int)
            assert "messages" in h and isinstance(h["messages"], list)
            steps.add(h["step"])
        # assert len(steps) == len(history), "Duplicate step names in history"
