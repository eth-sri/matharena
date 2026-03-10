import json
import threading
import time
from typing import Any
import os

from loguru import logger

from matharena.api_client import APIClient
from matharena.solvers import SolverResponse

class BaseAgent:
    """
    An abstract agent that solves a single math problem instance by using one or more APIClients.

    batch_idx: the index of this problem in the batch handled by the AgentPool.
    solver_config: the full solver config, including model_config and scaffold_config.
    default_prompt_template: the prompt template
        (the "instruction" from the competition config + {problem} template)
    default_api_client_args: the kwargs for the APIClient constructor, i.e., all kwargs
        stated in model_config + "tools" and "max_tool_calls" from the competition config.
        Tools is a list of pairs (function, tool_spec) where function is None for responses API
        The agent can override these when creating its own APIClient(s).
    """

    def __init__(self, batch_idx, problem_idx, run_idx, solver_config, 
                 default_prompt_template=None, default_api_client_args=None):
        self.batch_idx = batch_idx
        self.problem_idx = problem_idx
        self.run_idx = run_idx
        self.bi = batch_idx  # short alias
        self.solver_config = solver_config
        self.default_prompt_template = default_prompt_template
        self.default_api_client_args = default_api_client_args
        self._lock = threading.Lock()
        self.has_finished = False
        self.RUN_ID_FULL = None
        
        self.checkpoint_base_dir = "agent_checkpoints"

    def _start_run(self, stmt: str):
        """
        Starts the run:
            - Resets the state to be returned.
            - Sets the 1st of 2 entries in the final conversation.
        """
        self.stmt = stmt
        logger.debug(f"[{self.bi}] Starting agent run for problem: {stmt[:50]}...")
        self.conversation = [{"role": "user", "content": stmt}, {"role": "assistant", "content": "TODO"}]
        self.detailed_cost = {"cost": 0, "input_tokens": 0, "output_tokens": 0, "time": 0}
        self.history = []

    def _query(self, client: APIClient, query: list[dict[str, Any]], ignore_tool_calls: bool = False):
        """
        A wrapper that runs a single query (conversation) via the given APIClient and updates the cost state.
        Queries should add user/developer messages in clean format or reuse message blocks from same client.
        """
        start_time = time.time()
        ret = list(
            client.run_queries(
                [query], no_tqdm=True, custom_indices=[self.batch_idx], ignore_tool_calls=ignore_tool_calls
            )
        )
        _, conversation, detailed_cost = ret[0]
        with self._lock:
            self.detailed_cost["cost"] += detailed_cost["cost"]
            self.detailed_cost["input_tokens"] += detailed_cost["input_tokens"]
            self.detailed_cost["output_tokens"] += detailed_cost["output_tokens"]
            self.detailed_cost["time"] += time.time() - start_time
        return conversation

    def _add_history(self, step: str, timestep: int, conversation: Any, **kwargs):
        """
        Adds an entry to the history.
        """
        entry = {
            "step": step,
            "timestep": timestep,
            "messages": conversation,
        }
        entry.update(kwargs)
        with self._lock:
            self.history.append(entry)

    def _end_run(self, final_response) -> SolverResponse:
        """
        Ends the run:
            - Sets the 2nd of 2 entries in the final conversation.
            - Returns a SolverResponse object with batch index.
        """
        logger.debug(f"[{self.bi}] Ending agent run for problem: {self.stmt[:50]}...")
        if isinstance(final_response, str):
            self.conversation[1]["content"] = final_response
        else:
            self.conversation = final_response

        if len(self.conversation[-1]["content"]) == 0 or self.conversation[-1]["role"] != "assistant":
            logger.warning(f"[{self.bi}] Final conversation appears broken.")
            self.conversation.append({"role": "assistant", "content": "The model did not return an answer."})

        self.has_finished = True
        self._save_checkpoint()
        return SolverResponse(
            idx=self.batch_idx, conversation=self.conversation, detailed_cost=self.detailed_cost, history=self.history
        )
    
    def _find_full_run_id(self):
        if self.RUN_ID is None:
            self.RUN_ID = f"agent_b{self.batch_idx}_p{self.problem_idx}"
        if self.RUN_ID_FULL is not None:
            return
        
        indices_to_go = self.run_idx + 1
        current_index = 0
        while indices_to_go > 0:
            run_id = self.RUN_ID + f"_r{current_index}"
            checkpoint_path = f"{self.checkpoint_base_dir}/{run_id}.json"
            if not os.path.exists(checkpoint_path):
                indices_to_go -= 1
            else:
                with open(checkpoint_path, "r") as f:
                    checkpoint = json.load(f)
                    if not checkpoint.get("has_finished", True):
                        indices_to_go -= 1
            current_index += 1
        self.RUN_ID_FULL = self.RUN_ID + f"_r{current_index - 1}"

    def _load_checkpoint_if_exists(self) -> None:
        """
        Loads a checkpoint of the current detailed cost and history from a file.
        """
        if self.RUN_ID_FULL is None:
            self._find_full_run_id()

        checkpoint_path = f"{self.checkpoint_base_dir}/{self.RUN_ID_FULL}.json"
        os.makedirs(self.checkpoint_base_dir, exist_ok=True)
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
                self.detailed_cost = checkpoint["detailed_cost"]
                self.history = checkpoint["history"]
                log = f"[{self.bi}] Loaded checkpoint from {checkpoint_path}! Will skip the following steps:\n"
                for entry in self.history:
                    log += f"    - Step {entry['step']} at timestep {entry['timestep']}\n"
                logger.info(log)
        except FileNotFoundError:
            logger.info(f"[{self.bi}] No checkpoint found at {checkpoint_path}, starting fresh.")

    def _save_checkpoint(self) -> None:
        """
        Saves a checkpoint of the current detailed cost and history to a file.
        """
        if self.RUN_ID_FULL is None:
            self._find_full_run_id()
        checkpoint_path = f"{self.checkpoint_base_dir}/{self.RUN_ID_FULL}.json"
        logger.info(f"[{self.bi}] Saving checkpoint to {checkpoint_path}.")
        with open(checkpoint_path, "w") as f:
            json.dump(
                {
                    "detailed_cost": self.detailed_cost,
                    "history": self.history,
                    "has_finished": self.has_finished
                },
                f,
                indent=4,
            )

    def _history_has_step(self, step: str) -> bool:
        """
        Checks if a step exists in history.
        """

        for entry in self.history:
            if entry["step"] == step:
                return True
        return False

    def get_history_step(self, step: str) -> dict[str, Any]:
        """
        Retrieves a history entry by step name.
        """

        for entry in self.history:
            if entry["step"] == step:
                return entry
        raise ValueError(f"No history entry found for step {step}.")

    def _get_convo_from_history(self, step: str) -> dict[str, Any]:
        """
        Retrieves a conversation from history by step name.
        """

        for entry in self.history:
            if entry["step"] == step:
                return entry["messages"]
        raise ValueError(f"No history entry found for step {step}.")

    def solve(self, stmt: str) -> SolverResponse:
        """
        Solves a single problem statement.

        Args:
            stmt (str): A problem statement as text.

        Returns:
            SolverResponse: A SolverResponse object containing:
             - index: set to 0 here
             - conversation: the conversation array, for agents it should just have 2 blocks: "user" and "assistant"
             - detailed_cost: agent must report detailed cost info: cost, in/out tokens, time
             - history: a list of steps, where each step corresponds to one conversation:
                 - "step": unique string id
                 - "timestep": the time at which this step happened (for visualization)
                 - "messages": the full conversation in this step
                 - any extra debug keys
                 The agent decides what to report as the final response in the conversation.
        """
        raise NotImplementedError("Subclasses should implement this method.")
