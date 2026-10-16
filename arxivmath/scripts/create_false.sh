#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
CONTAINER_RUNTIME=docker
source "$REPO_ROOT/scripts/lib/container_runtime.sh"
parse_container_runtime "$@"
set -- "${CONTAINER_ARGS[@]}"
MODE="${1:-screen}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -ne 0 || ! "$MODE" =~ ^(screen|prepare|audit)$ ]]; then
  echo "Usage: bash arxivmath/scripts/create_false.sh [screen|prepare|audit] [--container-runtime docker|podman]" >&2
  exit 2
fi

PAPER_ROOT="${ARXIV_FALSE_PAPER_ROOT:-arxivmath/paper}"
MODEL_CONFIG=()
if [[ -n "${ARXIV_FALSE_MODEL_CONFIG:-}" ]]; then MODEL_CONFIG=(--model-config "$ARXIV_FALSE_MODEL_CONFIG"); fi
SOURCE_CACHE="${ARXIV_SOURCE_CACHE:-arxivmath/source_cache}"
SCOPE=()
OVERWRITE=()
SCREEN_OVERWRITE=()
INVESTIGATION_BUDGET=()
VERIFICATION_BUDGET=()
if [[ -n "${ARXIV_FALSE_MAX_PAPERS:-}" ]]; then SCOPE+=(--max-papers "$ARXIV_FALSE_MAX_PAPERS"); fi
if [[ "${ARXIV_FALSE_MODEL_OVERWRITE:-0}" == "1" ]]; then OVERWRITE+=(--overwrite); fi
if [[ "${ARXIV_FALSE_SCREEN_OVERWRITE:-0}" == "1" ]]; then SCREEN_OVERWRITE+=(--overwrite); fi
if [[ -n "${ARXIV_FALSE_INVESTIGATION_MAX_COST:-}" ]]; then INVESTIGATION_BUDGET+=(--max-cost "$ARXIV_FALSE_INVESTIGATION_MAX_COST"); fi
if [[ -n "${ARXIV_FALSE_VERIFICATION_MAX_COST:-}" ]]; then VERIFICATION_BUDGET+=(--max-cost "$ARXIV_FALSE_VERIFICATION_MAX_COST"); fi
COMMON=(--false --paper-root "$PAPER_ROOT" --source-cache "$SOURCE_CACHE" "${SCOPE[@]}")
PREPARE_STATUS=0

if [[ "$MODE" == "screen" || "$MODE" == "prepare" ]]; then
  uv run python arxivmath/scripts/source/screen_abstracts.py --false \
    --container-runtime "$CONTAINER_RUNTIME" \
    "${MODEL_CONFIG[@]}" --paper-root "$PAPER_ROOT" "${SCOPE[@]}" "${SCREEN_OVERWRITE[@]}"
  if [[ "$MODE" == "screen" ]]; then
    echo "Screen complete. Next, when ready: bash arxivmath/scripts/create_false.sh prepare --container-runtime $CONTAINER_RUNTIME"
    exit 0
  fi
  uv run python arxivmath/scripts/source/download_sources.py --false \
    --paper-root "$PAPER_ROOT" --source-cache "$SOURCE_CACHE" \
    "${SCOPE[@]}"
  uv run python arxivmath/scripts/source/investigate_sources.py "${COMMON[@]}" \
    --container-runtime "$CONTAINER_RUNTIME" \
    "${MODEL_CONFIG[@]}" --batch-size "${ARXIV_FALSE_BATCH_SIZE:-2}" \
    --max-source-tokens "${ARXIV_FALSE_MAX_SOURCE_TOKENS:-750000}" "${OVERWRITE[@]}" "${INVESTIGATION_BUDGET[@]}" || {
      PREPARE_STATUS=$?
      if [[ "$PREPARE_STATUS" != "1" ]]; then exit "$PREPARE_STATUS"; fi
      echo "Source investigation is incomplete; continuing verification of completed candidates." >&2
    }
  uv run python arxivmath/scripts/source/verify_questions.py "${COMMON[@]}" \
    --container-runtime "$CONTAINER_RUNTIME" \
    "${MODEL_CONFIG[@]}" --batch-size "${ARXIV_FALSE_BATCH_SIZE:-2}" "${OVERWRITE[@]}" "${VERIFICATION_BUDGET[@]}" || PREPARE_STATUS=$?
fi
if [[ "$MODE" == "audit" ]]; then
  uv run python arxivmath/scripts/source/audit_pipeline.py "${COMMON[@]}"
else
  echo "Review: uv run python arxivmath/app.py --false --paper-root '$PAPER_ROOT'"
fi
exit "$PREPARE_STATUS"
