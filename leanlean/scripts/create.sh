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
october_selection="$repo_root/configs/leanlean/october-2026.json"
selection="${LEANLEAN_SELECTION:-$october_selection}"
bundle="$work_root/run/bundle"
dataset=""
if [[ "$mode" == "prepare" && "$(realpath -m "$selection")" == "$october_selection" ]]; then
    dataset="$repo_root/data/leanlean/oct26"
    if [[ "$work_root/" == "$dataset/"* ]]; then
        echo "LEANLEAN_WORK_ROOT must be outside $dataset" >&2
        exit 1
    fi
    if [[ -L "$dataset" && "$(realpath -m "$dataset")" != "$bundle" ]]; then
        echo "The October dataset path already links to another bundle: $dataset" >&2
        exit 1
    fi
    if [[ -e "$dataset" && ! -f "$dataset/manifest.json" ]]; then
        echo "The October dataset path already exists without a bundle manifest: $dataset" >&2
        exit 1
    fi
fi
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

if [[ -n "$dataset" ]]; then
    mkdir -p "$(dirname "$dataset")"
    if [[ ! -e "$dataset" && ! -L "$dataset" ]]; then
        ln -sT "$bundle" "$dataset"
    fi
    if [[ -L "$dataset" ]]; then
        bundle="$dataset"
    else
        echo "Kept the existing October dataset at $dataset."
    fi
fi

echo "Preparation is complete. Review $bundle/index.csv"
echo "Bundle: $bundle"
echo "Use the verified bundle to prepare an evaluation image."
