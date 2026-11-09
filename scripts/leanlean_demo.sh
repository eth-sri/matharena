#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
    echo "Usage: bash scripts/leanlean_demo.sh /absolute/path/to/demo-workspace" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="$(realpath -m "$1")"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${repo_root}"

uv run --extra leanlean python -m leanlean.palomar_source \
    --selection configs/leanlean/demo.json \
    --output "${run_root}/snapshot" --workers 4

uv run --extra leanlean python -m leanlean.strip \
    --snapshot "${run_root}/snapshot" --output "${run_root}/run" \
    --workers 4 --cpus 8 --memory 64g --storage-root "${run_root}/podman"
