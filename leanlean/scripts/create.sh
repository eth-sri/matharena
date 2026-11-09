#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

mode="${1:-fetch}"
if [[ $# -gt 0 ]]; then
    shift
fi
if [[ $# -ne 0 || ( "$mode" != "fetch" && "$mode" != "prepare" ) ]]; then
    echo "Usage: bash leanlean/scripts/create.sh [fetch|prepare]" >&2
    exit 2
fi

work_root="$(realpath -m "${LEANLEAN_WORK_ROOT:-leanlean/work/oct26}")"
selection="${LEANLEAN_SELECTION:-configs/leanlean/october-2026.json}"
workers="${LEANLEAN_WORKERS:-6}"
cpus="${LEANLEAN_CPUS:-8}"
memory="${LEANLEAN_MEMORY:-64g}"
timeout="${LEANLEAN_TIMEOUT:-21600}"
storage_root="${LEANLEAN_PODMAN_STORAGE_ROOT:-${work_root}/podman}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "LeanLean source preparation"
echo "  mode: $mode"
echo "  selection: $selection"
echo "  workspace: $work_root"
echo "  workers: $workers"

uv run --extra leanlean python -m leanlean.palomar_source \
    --selection "$selection" --output "$work_root/snapshot" --workers "$workers"

if [[ "$mode" == "fetch" ]]; then
    echo "Source fetching is complete. Inspect $work_root/snapshot/index.csv"
    echo "Next, with the same environment: bash leanlean/scripts/create.sh prepare"
    exit 0
fi

echo "  container limits: $cpus CPUs, $memory RAM"
echo "  Podman storage: $storage_root"
uv run --extra leanlean python -m leanlean.strip \
    --snapshot "$work_root/snapshot" --output "$work_root/run" \
    --workers "$workers" --cpus "$cpus" --memory "$memory" --timeout "$timeout" \
    --storage-root "$storage_root"

echo "Preparation is complete. Review $work_root/run/bundle/index.csv"
echo "Bundle: $work_root/run/bundle"
echo "Use the verified bundle to prepare an evaluation image."
