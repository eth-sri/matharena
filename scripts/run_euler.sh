CONTAINER_RUNTIME=docker
source "$(dirname "${BASH_SOURCE[0]}")/lib/container_runtime.sh"
parse_container_runtime "$@" || exit $?
if (( ${#CONTAINER_ARGS[@]} != 0 )); then
  echo "Usage: $0 [--container-runtime docker|podman]" >&2
  exit 2
fi
MODELS=(
  "gemini/gemini-31-pro"
  "openai/gpt-54"
  "anthropic/opus_46"
  "moonshot/k25"
)

for model in "${MODELS[@]}"; do
  model_name=${model//\//_}
  python -u scripts/run.py --comp euler/euler --models "$model" --n 2 --container-runtime "$CONTAINER_RUNTIME" > "${model_name}.out" 2>&1 &
done

wait
