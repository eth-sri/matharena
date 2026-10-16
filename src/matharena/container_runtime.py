"""Container backend selection for execution entry points, independent of models."""

from argparse import ArgumentParser

CONTAINER_RUNTIMES = ("docker", "podman")


def validate_container_runtime(runtime: str) -> str:
    if not isinstance(runtime, str):
        raise TypeError("container_runtime must be a string")
    runtime = runtime.strip().lower()
    if runtime not in CONTAINER_RUNTIMES:
        raise ValueError("container_runtime must be 'docker' or 'podman'")
    return runtime


def add_container_runtime_argument(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--container-runtime", choices=CONTAINER_RUNTIMES, default="docker",
        help="Container backend for harness execution (default: docker)",
    )
