#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONTAINER_RUNTIME=docker
source "$REPO_ROOT/scripts/lib/container_runtime.sh"
parse_container_runtime "$@"
set -- "${CONTAINER_ARGS[@]}"

MODE="${1:-screen}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -ne 0 || ( "$MODE" != "screen" && "$MODE" != "prepare" ) ]]; then
  echo "Usage: bash arxivmath/scripts/create.sh [screen|prepare] [--container-runtime docker|podman]" >&2
  exit 2
fi

PAPER_ROOT="${ARXIVMATH_PAPER_ROOT:-arxivmath/paper}"
SCREEN_CONFIG="${ARXIVMATH_SCREEN_CONFIG:-openai/gpt-6-astra}"
INVESTIGATOR_CONFIG="${ARXIVMATH_INVESTIGATOR_CONFIG:-openai/gpt-6-astra}"
VERIFIER_CONFIG="${ARXIVMATH_VERIFIER_CONFIG:-openai/gpt-6-astra}"

if [[ -n "${ARXIV_SOURCE_CACHE:-}" ]]; then
  SOURCE_CACHE="$ARXIV_SOURCE_CACHE"
elif [[ -d "/userdata/${USER:-}" ]]; then
  SOURCE_CACHE="/userdata/${USER}/matharena-arxiv-source"
elif [[ -d "/scratch/userdata/${USER:-}" ]]; then
  SOURCE_CACHE="/scratch/userdata/${USER}/matharena-arxiv-source"
else
  SOURCE_CACHE="arxivmath/source_cache"
fi
REQUEST_LOG_DIR="${MATHARENA_REQUEST_LOG_DIR:-${SOURCE_CACHE%/}/request-logs}"
export MATHARENA_REQUEST_LOG_DIR="$REQUEST_LOG_DIR"

OVERWRITE_ARGS=()
SCREEN_OVERWRITE_ARGS=()
SOURCE_ARGS=()
AVAILABILITY_ARGS=()
SCOPE_ARGS=()
INVESTIGATION_BUDGET_ARGS=()
VERIFICATION_BUDGET_ARGS=()
SOURCE_LIMIT_ARGS=()
if [[ "${ARXIVMATH_SOURCE_OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi
if [[ "${ARXIVMATH_SCREEN_OVERWRITE:-0}" == "1" ]]; then
  SCREEN_OVERWRITE_ARGS+=(--overwrite)
fi
if [[ "${ARXIVMATH_ALLOW_UNRESOLVED_INCLUDES:-0}" == "1" ]]; then
  SOURCE_ARGS+=(--allow-unresolved-includes)
fi
if [[ "${ARXIVMATH_ALLOW_SOURCE_UNAVAILABLE:-0}" == "1" ]]; then
  AVAILABILITY_ARGS+=(--allow-source-unavailable)
fi
if [[ -n "${ARXIVMATH_MAX_PAPERS:-}" ]]; then
  SCOPE_ARGS+=(--max-papers "$ARXIVMATH_MAX_PAPERS")
fi
if [[ -n "${ARXIVMATH_INVESTIGATION_MAX_COST:-}" ]]; then
  INVESTIGATION_BUDGET_ARGS+=(--max-cost "$ARXIVMATH_INVESTIGATION_MAX_COST")
fi
if [[ -n "${ARXIVMATH_VERIFICATION_MAX_COST:-}" ]]; then
  VERIFICATION_BUDGET_ARGS+=(--max-cost "$ARXIVMATH_VERIFICATION_MAX_COST")
fi
if [[ -n "${ARXIVMATH_MAX_SOURCE_TOKENS:-}" ]]; then
  SOURCE_LIMIT_ARGS+=(--max-source-tokens "$ARXIVMATH_MAX_SOURCE_TOKENS")
fi

echo "ArXivMath abstract-screened, single-question source pipeline"
echo "  mode: $MODE"
echo "  paper root: $PAPER_ROOT"
echo "  source cache: $SOURCE_CACHE"
echo "  request logs: $REQUEST_LOG_DIR"
echo "  model backend: harness_wrapper via the normal model configs"

if [[ "$MODE" == "screen" || "$MODE" == "prepare" ]]; then
  echo "  abstract screen: $SCREEN_CONFIG"

  uv run python arxivmath/scripts/source/screen_abstracts.py \
    --model-config "$SCREEN_CONFIG" \
    --container-runtime "$CONTAINER_RUNTIME" \
    --paper-root "$PAPER_ROOT" \
    "${SCOPE_ARGS[@]}" \
    "${SCREEN_OVERWRITE_ARGS[@]}"

  if [[ "$MODE" == "screen" ]]; then
    echo "Abstract screening is complete."
    echo "Inspect the accepted-paper count before starting source investigation."
    echo "Next: nice -n 7 bash arxivmath/scripts/create.sh prepare --container-runtime $CONTAINER_RUNTIME"
    exit 0
  fi

  echo "  source investigator: $INVESTIGATOR_CONFIG"
  echo "  independent verifier: $VERIFIER_CONFIG"

  uv run python arxivmath/scripts/source/download_sources.py \
    --paper-root "$PAPER_ROOT" \
    --source-cache "$SOURCE_CACHE" \
    "${SCOPE_ARGS[@]}" \
    "${SOURCE_ARGS[@]}" \
    "${AVAILABILITY_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}"

  uv run python arxivmath/scripts/source/investigate_sources.py \
    --model-config "$INVESTIGATOR_CONFIG" \
    --container-runtime "$CONTAINER_RUNTIME" \
    --paper-root "$PAPER_ROOT" \
    --source-cache "$SOURCE_CACHE" \
    "${SCOPE_ARGS[@]}" \
    "${AVAILABILITY_ARGS[@]}" \
    "${SOURCE_LIMIT_ARGS[@]}" \
    "${INVESTIGATION_BUDGET_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}"

  uv run python arxivmath/scripts/source/verify_questions.py \
    --model-config "$VERIFIER_CONFIG" \
    --container-runtime "$CONTAINER_RUNTIME" \
    --paper-root "$PAPER_ROOT" \
    --source-cache "$SOURCE_CACHE" \
    "${SCOPE_ARGS[@]}" \
    "${AVAILABILITY_ARGS[@]}" \
    "${VERIFICATION_BUDGET_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}"
fi

echo "Source preparation is complete."
echo "Review questions: uv run python arxivmath/app.py --paper-root $PAPER_ROOT"
echo "Audit: uv run python arxivmath/scripts/source/audit_pipeline.py --paper-root $PAPER_ROOT --source-cache $SOURCE_CACHE ${AVAILABILITY_ARGS[*]}"
