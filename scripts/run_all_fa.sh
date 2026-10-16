CONTAINER_RUNTIME=docker
source "$(dirname "${BASH_SOURCE[0]}")/lib/container_runtime.sh"
parse_container_runtime "$@" || exit $?
set -- "${CONTAINER_ARGS[@]}"
if (( $# != 1 )); then
  echo "Usage: $0 MODEL [--container-runtime docker|podman]" >&2
  exit 2
fi
MODEL=$1
DEFAULT_N=4

COMPS=(
  # "arxiv/may"
  # "arxiv/june"
  "arxiv/august"
  # "arxiv_false/may"
  # "arxiv_false/june"
  "arxiv_false/august"
)

# Per-comp n overrides
declare -A N_VALUES=(
  # add more overrides here if needed
  # ["arxiv/april"]=3
  # ["arxiv_false/april"]=2
  ["arxiv/may"]=3
  ["arxiv_false/may"]=2
  ["arxiv/june"]=3
  ["arxiv_false/june"]=2
  ["arxiv/august"]=1
  ["arxiv_false/august"]=1
)

COMP_N_OVERRIDES=()
for comp in "${COMPS[@]}"; do
  if [[ -n "${N_VALUES[$comp]}" ]]; then
    COMP_N_OVERRIDES+=("${comp}=${N_VALUES[$comp]}")
  fi
done

echo "Running on comps: ${COMPS[*]} with model $MODEL (default n=$DEFAULT_N, overrides: ${COMP_N_OVERRIDES[*]})"
uv run python scripts/run.py \
  --comp "${COMPS[@]}" \
  --models "$MODEL" \
  --container-runtime "$CONTAINER_RUNTIME" \
  --n "$DEFAULT_N" \
  --comp-n "${COMP_N_OVERRIDES[@]}"

for comp in "${COMPS[@]}"; do
  # August datasets use dedicated judges instead of the annotation checker.
  if [[ "$comp" != "arxiv/august" && "$comp" != "arxiv_false/august" ]]; then
    uv run python scripts/curation/check.py --comp "$comp" --model-config gemini/gemini-38-flash-low
  fi
done

for comp in "${COMPS[@]}"; do
  if [[ "$comp" == arxiv_false/* ]]; then
    uv run python scripts/judge/judge.py --comp "$comp" --models "$MODEL" --container-runtime "$CONTAINER_RUNTIME"
  fi
done
