#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "Usage: $0 STRIP_RUN BUNDLE OUTPUT" >&2
    exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${repo_root}"
storage_args=()
if [[ -n "${LEANLEAN_PODMAN_STORAGE_ROOT:-}" ]]; then
    storage_args=(--storage-root "$LEANLEAN_PODMAN_STORAGE_ROOT")
fi
uv run --no-sync python -m leanlean.prepare_eval \
    --run "$1" --bundle "$2" --output "$3" "${storage_args[@]}"
