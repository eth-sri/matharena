CONTAINER_RUNTIME=docker
source "$(dirname "${BASH_SOURCE[0]}")/lib/container_runtime.sh"
parse_container_runtime "$@" || exit $?
set -- "${CONTAINER_ARGS[@]}"
if (( $# != 1 )); then
  echo "Usage: $0 MODEL [--container-runtime docker|podman]" >&2
  exit 2
fi
MODEL=$1
COMPS=(
  "kangaroo/kangaroo_2025_1-2"
  "kangaroo/kangaroo_2025_11-12"
  "kangaroo/kangaroo_2025_3-4"
  "kangaroo/kangaroo_2025_5-6"
  "kangaroo/kangaroo_2025_7-8"
  "kangaroo/kangaroo_2025_9-10"
)

echo "Running on comps: ${COMPS[*]} with model $MODEL"
uv run python3 scripts/run.py \
  --container-runtime "$CONTAINER_RUNTIME" \
  --comp "${COMPS[@]}" \
  --n 2 \
  --models "$MODEL"
