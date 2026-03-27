import re, yaml, os
from concurrent.futures import ThreadPoolExecutor

from matharena.api_client import APIClient
from matharena.solvers.judges.base_judge import BaseJudge
from matharena.utils import normalize_conversation
from matharena.tools.code_execution import execute_code


class NormalizeJudge(BaseJudge):
    def __init__(self, batch_idx, problem_idx, run_idx, solver_config):
        super().__init__(batch_idx, problem_idx, run_idx, solver_config)
        self.prepare_clients()
        self.RUN_ID = None

    def prepare_clients(self):
        self.clients = []
        self.client_names = []
        for model_config_name in self.solver_config["model_configs"]:
            model_config = yaml.safe_load(open(os.path.join("configs", "models", model_config_name + ".yaml")))
            for key in self.solver_config.get("api_client_remove_keys", []):
                model_config.pop(key, None)

            tools = []
            for tool_name in self.solver_config["enabled_tools"]:
                if tool_name == "execute_code":
                    spec = self.solver_config["tool_specs"]["execute_code"]
                    tools.append((execute_code, spec["tool_spec"]))
            model_config["tools"] = tools
            model_config["max_tool_calls"] = self.solver_config["max_tool_calls"]

            client = APIClient(**model_config)
            self.clients.append(client)
            self.client_names.append(model_config_name)

        self.main_model_config = self.solver_config["model_config"]
        for key in self.solver_config.get("api_client_remove_keys", []):
            self.main_model_config.pop(key, None)
        
        self.main_client = APIClient(**self.main_model_config)

    def do_iteration(self, points):
        points_not_none = [p for p in points if p is not None]
        if len(points_not_none) == 0:
            return True
        if abs(min(points_not_none) - max(points_not_none)) <= self.solver_config.get("iterate_diff", 0):
            return False
        return True
    
    def get_final_grade(self, points):
        points_not_none = [p for p in points if p is not None]
        if self.solver_config.get("aggregate_points_by") == "max":
            return max(points_not_none)
        elif self.solver_config.get("aggregate_points_by") == "average":
            return round(sum(points_not_none) / len(points_not_none))
        elif self.solver_config.get("aggregate_points_by", "min") == "min":
            return min(points_not_none)
        else:
            raise ValueError(f"Invalid aggregate_points_by: {self.solver_config.get('aggregate_points_by')}")
        
    def extract_and_run(self, prompt, iteration=0):
        def run_client(idx_client):
            i, client = idx_client
            start_time = __import__('time').time()
            ret = list(client.run_queries(
                [[{"role": "user", "content": prompt}]],
                no_tqdm=True, custom_indices=[self.batch_idx],
                ignore_tool_calls=False,
            ))
            _, raw_conversation, query_detailed_cost = ret[0]
            elapsed = __import__('time').time() - start_time
            with self._lock:
                self.detailed_cost["cost"] += query_detailed_cost["cost"]
                self.detailed_cost["input_tokens"] += query_detailed_cost["input_tokens"]
                self.detailed_cost["output_tokens"] += query_detailed_cost["output_tokens"]
                self.detailed_cost["time"] += elapsed
            conversation = normalize_conversation(raw_conversation)
            text = conversation[-1]["content"]
            m_points = re.search(r"<points>\s*([0-9]+)\s*</points>", text, flags=re.IGNORECASE | re.DOTALL)
            points = max(0, min(int(m_points.group(1).strip()), 7)) if m_points else None
            return i, conversation, text, points, query_detailed_cost["cost"]

        outputs = [None] * len(self.clients)
        points = [None] * len(self.clients)
        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            results = list(executor.map(run_client, enumerate(self.clients)))

        for i, conversation, text, point, cost in sorted(results, key=lambda result: result[0]):
            self._add_history(f"Iteration {iteration}: {self.client_names[i]}", i + 1 + len(self.clients) * iteration, conversation)
            cost_by_judge = self.detailed_cost.setdefault("cost_by_judge_model", {})
            cost_by_judge[self.client_names[i]] = cost_by_judge.get(self.client_names[i], 0.0) + cost
            # remove ```xml from text
            if text.startswith("```xml"):
                text = text[len("```xml"):].strip()
                text = text[:-len("```")].strip() if text.endswith("```") else text
            outputs[i] = text
            points[i] = point
        outputs_as_str = "\n\n".join(f"### Output from judge {i + 1} ###\n{text}" for i, text in enumerate(outputs))
        return points, outputs_as_str

    def solve(
        self,
        problem_statement: str,
        guidelines: str,
        ground_truth_solutions: list[str],
        student_answer: str,
        original_problem_statement: str = "",
    ):
        self._start_run(problem_statement)
        if len(ground_truth_solutions) == 0:
            gt_text = "None provided."
        elif len(ground_truth_solutions) == 1:
            gt_text = ground_truth_solutions[0]
        else:
            gt_text = "\n\n".join(
                f"### Ground truth proof {i + 1} ###\n{proof}" for i, proof in enumerate(ground_truth_solutions)
            )

        prompt_normalize = self.solver_config["normalize_prompt"].format(
            problem_statement=problem_statement,
            original_problem_statement=original_problem_statement,
            student_answer=student_answer,
        )

        additional_info = {

        }

        conversation = normalize_conversation(self._query(self.main_client, [{"role": "user", "content": prompt_normalize}]))
        student_answer = conversation[-1]["content"]
        additional_info["normalized_answer"] = student_answer
        self._add_history("normalize", 1, conversation)

        prompt = self.solver_config["judge_prompt"].format(
            problem_statement=problem_statement,
            original_problem_statement=original_problem_statement,
            guidelines=guidelines,
            student_answer=student_answer,
            ground_truth_solutions=gt_text,
        )
        
        points, outputs_as_str = self.extract_and_run(prompt)

        additional_info["initial_judge_outputs"] = outputs_as_str

        for idx in range(self.solver_config.get("iterations_judgment", 0)):
            if not self.do_iteration(points):
                break
            new_prompt = self.solver_config["judge_prompt_repetition"].format(
                judge_prompt=prompt,
                other_judgments=outputs_as_str,
            )
            points, outputs_as_str = self.extract_and_run(new_prompt, iteration=idx + 1)
            additional_info[f"judge_outputs_iteration_{idx + 1}"] = outputs_as_str

        if all(p is None for p in points):
            return self._end_run(None, outputs_as_str)
        final_points = self.get_final_grade(points)

        self._save_checkpoint()
       
        return self._end_run(final_points, outputs_as_str, additional_info)
