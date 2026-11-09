from typing import Any, Protocol
from pathlib import Path


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, command: str, cwd: str = "") -> dict[str, str]: ...

    def get_template_vars(self) -> dict[str, Any]: ...


class Instance(Protocol):
    """Protocol for task instances."""

    instance_id: str
    repo: str
    task: str
    patch: str # The solution of the instance

    def get_dir(self, base_dir: Path) -> Path: ...

    def setup(self, env_config: dict[str, Any]) -> Environment: ...

    def solve(self, patch_diff: str, base_dir: Path, run_id: int) -> bool: ...

    def get_submission_gate(self) -> dict[str, Any]:
        return {}

    def validate_submission(self, env: Environment) -> dict[str, Any]:
        return {"passed": True, "command": "", "returncode": 0, "output": ""}

    def build_submission_feedback(self, validation_result: dict[str, Any]) -> str:
        return (
            "Your previous submission failed validation.\n"
            f"Output:\n{validation_result.get('output', '')}\n"
            "Please fix and submit again."
        )

class Benchmark(Protocol):
    """Protocol for benchmarks."""

    config: Any

    def get_single_instance(self, instance_id: str) -> Instance: ...

    def get_instances(self) -> list[Instance]: ...

    def solve(self, patch_diff: str, base_dir: Path, run_id: int) -> dict[str, bool]: ...


class Model(Protocol):
    """Protocol for language models."""

    config: Any
    cost: float
    n_calls: int
    responses: list[dict[str, Any]]

    def serve(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_openai_args(self) -> dict[str, Any]:
        return {"model": None, "api_key": None, "base_url": None}

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...

    def get_template_vars(self) -> dict[str, Any]: ...

    def get_name(self) -> str: ...

class Generator(Protocol):
    """Protocol for generators."""

    model: Model
    env: Environment
    messages: list[dict[str, str]]
    config: Any

    def run(self, task: str, **kwargs) -> tuple[str, str]: ...

class Planner(Protocol):
    """Protocol for planners."""

    config: Any

    def plan(self, env: Environment, model: Model, instance: Instance) -> None: ...

    def update_plan(self, instance: Instance, traces: list[dict], result: str, base_dir: Path) -> None: ...

    def get_template_vars(self) -> dict[str, Any]: ...

    def get_name(self) -> str: ...

__all__ = [
    "Model",
    "Environment",
    "Instance",
    "Benchmark",
    "Generator",
    "Planner",
]
