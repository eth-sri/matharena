#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0 || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TRAIN_ROOT}/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${TRAIN_ROOT}/configs/grpo_qwen35_2b.env}"

if [[ "${SOURCE_CONFIG:-true}" == "true" && -f "${CONFIG_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
  set +a
fi
if [[ -f "${TRAIN_ROOT}/configs/local.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${TRAIN_ROOT}/configs/local.env"
  set +a
fi

CONTAINER_RUNTIME=docker
source "$REPO_ROOT/scripts/lib/container_runtime.sh"
parse_container_runtime "$@"
set -- "${CONTAINER_ARGS[@]}"
if (( $# > 1 )); then
  echo "Usage: $0 [COMPETITION] [--container-runtime docker|podman]" >&2
  exit 2
fi

: "${SCRATCH:?SCRATCH must be set on CSCS}"

export HF_HOME="${HF_HOME:-${SCRATCH}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTHONUNBUFFERED=1
unset SETUPTOOLS_USE_DISTUTILS
export TORCH_CUDNN_V8_API_DISABLED="${TORCH_CUDNN_V8_API_DISABLED:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
unset VLLM_USE_V1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "Unsetting PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}; vLLM memory pool is incompatible with expandable_segments." >&2
  unset PYTORCH_CUDA_ALLOC_CONF
fi
EVAL_VLLM_ATTENTION_BACKEND="${EVAL_VLLM_ATTENTION_BACKEND:-${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}}"
unset VLLM_VERSION VLLM_FLASH_ATTN_SRC_DIR VLLM_ATTENTION_BACKEND
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
fi

python - <<'PY'
import sys

required = ("datasets", "openai", "matharena", "transformers", "vllm")
missing = []
for name in required:
    try:
        __import__(name)
    except ModuleNotFoundError:
        missing.append(name)

if missing:
    raise SystemExit(
        "Missing evaluator dependency/dependencies in the active Python env: "
        + ", ".join(missing)
        + f"\npython={sys.executable}\n"
        + "Run: sbatch --account=a0163 train/slurm/06_repair_verl_env_deps.sbatch\n"
        + "If repair still fails, recreate the env with: sbatch --account=a0163 train/slurm/00_setup_verl_env.sbatch"
    )

print(f"eval_python={sys.executable}")
import transformers
import vllm
print(f"eval_transformers={transformers.__version__} path={transformers.__file__}")
print(f"eval_vllm={vllm.__version__} path={vllm.__file__}")
PY

EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-${HF_EXPORT_DIR:-${SCRATCH}/hf_exports/${EXPERIMENT_NAME:-qwen35-2b-grpo-gh200}}}"
EVAL_SERVED_MODEL_NAME="${EVAL_SERVED_MODEL_NAME:-matharena/qwen3.5-2b-arxivmath-grpo}"
EVAL_MODEL_CONFIG="${EVAL_MODEL_CONFIG:-qwen/qwen3.5_2b_arxivmath_grpo}"
EVAL_COMP="${1:-${EVAL_COMP:-arxiv/march}}"
EVAL_N="${EVAL_N:-4}"
if [[ -z "${EVAL_NUM_SHARDS:-}" && -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  EVAL_NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT}"
fi
if [[ -z "${EVAL_SHARD_INDEX:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  EVAL_ARRAY_MIN="${SLURM_ARRAY_TASK_MIN:-0}"
  EVAL_SHARD_INDEX="$((SLURM_ARRAY_TASK_ID - EVAL_ARRAY_MIN))"
fi
EVAL_NUM_SHARDS="${EVAL_NUM_SHARDS:-1}"
EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-0}"
if ! [[ "${EVAL_NUM_SHARDS}" =~ ^[0-9]+$ ]] || (( EVAL_NUM_SHARDS < 1 )); then
  echo "EVAL_NUM_SHARDS must be a positive integer; got ${EVAL_NUM_SHARDS}" >&2
  exit 1
fi
if ! [[ "${EVAL_SHARD_INDEX}" =~ ^[0-9]+$ ]] || (( EVAL_SHARD_INDEX >= EVAL_NUM_SHARDS )); then
  echo "EVAL_SHARD_INDEX must be in [0, EVAL_NUM_SHARDS); got ${EVAL_SHARD_INDEX}/${EVAL_NUM_SHARDS}" >&2
  exit 1
fi
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs}"
EVAL_VLLM_HOST="${EVAL_VLLM_HOST:-127.0.0.1}"
EVAL_VLLM_PORT="${EVAL_VLLM_PORT:-8004}"
EVAL_GPUS_PER_TASK="${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_ON_NODE:-1}}"
EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-${EVAL_GPUS_PER_TASK}}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-131072}"
EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-64}"
EVAL_MAX_NUM_BATCHED_TOKENS="${EVAL_MAX_NUM_BATCHED_TOKENS:-65536}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.90}"
EVAL_VLLM_STARTUP_TIMEOUT="${EVAL_VLLM_STARTUP_TIMEOUT:-1800}"
EVAL_JOB_TAG="${SLURM_JOB_ID:-manual}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  EVAL_JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-array}}-${SLURM_ARRAY_TASK_ID}"
fi
EVAL_VLLM_LOG="${EVAL_VLLM_LOG:-train/logs/vllm-eval-${EVAL_JOB_TAG}.log}"
EVAL_REASONING_PARSER="${EVAL_REASONING_PARSER:-qwen3}"
EVAL_QWEN35_LANGUAGE_MODEL_ONLY="${EVAL_QWEN35_LANGUAGE_MODEL_ONLY:-auto}"

EVAL_REASONING_ARGS=()
if [[ -n "${EVAL_REASONING_PARSER}" ]]; then
  EVAL_REASONING_ARGS+=(--reasoning-parser "${EVAL_REASONING_PARSER}")
fi

EVAL_MODEL_TYPE="$(
  python - "${EVAL_MODEL_PATH}" <<'PY'
import sys
from transformers import AutoConfig

config = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
print(getattr(config, "model_type", ""))
PY
)"

if [[ "${EVAL_QWEN35_LANGUAGE_MODEL_ONLY}" == "auto" ]]; then
  case "${EVAL_MODEL_TYPE}" in
    qwen3_5|qwen3_5_moe)
      EVAL_QWEN35_LANGUAGE_MODEL_ONLY=true
      ;;
    qwen3_5_text|qwen3_5_moe_text)
      EVAL_QWEN35_LANGUAGE_MODEL_ONLY=false
      ;;
    *)
      EVAL_QWEN35_LANGUAGE_MODEL_ONLY=false
      ;;
  esac
fi

EVAL_LANGUAGE_MODEL_ONLY_ARGS=()
case "${EVAL_QWEN35_LANGUAGE_MODEL_ONLY}" in
  1|true|TRUE|yes|YES)
    EVAL_LANGUAGE_MODEL_ONLY_ARGS+=(--language-model-only)
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "EVAL_QWEN35_LANGUAGE_MODEL_ONLY must be auto, true, or false; got ${EVAL_QWEN35_LANGUAGE_MODEL_ONLY}" >&2
    exit 1
    ;;
esac

mkdir -p "${HF_HOME}" "$(dirname "${EVAL_VLLM_LOG}")" "${EVAL_OUTPUT_DIR}"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Starting vLLM server for ${EVAL_MODEL_PATH}"
echo "vLLM log: ${EVAL_VLLM_LOG}"
echo "vLLM tensor_parallel_size=${EVAL_TENSOR_PARALLEL_SIZE} visible_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "vLLM hf_model_type=${EVAL_MODEL_TYPE} language_model_only=${EVAL_QWEN35_LANGUAGE_MODEL_ONLY}"
echo "Waiting up to ${EVAL_VLLM_STARTUP_TIMEOUT}s for http://${EVAL_VLLM_HOST}:${EVAL_VLLM_PORT}/v1/models"
VLLM_LAUNCHER=(python -c 'from vllm.entrypoints.cli.main import main; raise SystemExit(main())')
"${VLLM_LAUNCHER[@]}" serve "${EVAL_MODEL_PATH}" \
  --served-model-name "${EVAL_SERVED_MODEL_NAME}" \
  --host "${EVAL_VLLM_HOST}" \
  --port "${EVAL_VLLM_PORT}" \
  --tensor-parallel-size "${EVAL_TENSOR_PARALLEL_SIZE}" \
  --dtype bfloat16 \
  --max-model-len "${EVAL_MAX_MODEL_LEN}" \
  --max-num-seqs "${EVAL_MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${EVAL_MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION}" \
  --attention-backend "${EVAL_VLLM_ATTENTION_BACKEND}" \
  "${EVAL_REASONING_ARGS[@]}" \
  "${EVAL_LANGUAGE_MODEL_ONLY_ARGS[@]}" \
  --trust-remote-code \
  ${EVAL_VLLM_EXTRA_ARGS:-} \
  >"${EVAL_VLLM_LOG}" 2>&1 &
VLLM_PID=$!
export VLLM_PID EVAL_VLLM_HOST EVAL_VLLM_PORT EVAL_VLLM_STARTUP_TIMEOUT EVAL_VLLM_LOG

python - <<PY
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

host = os.environ["EVAL_VLLM_HOST"]
port = os.environ["EVAL_VLLM_PORT"]
pid = int(os.environ["VLLM_PID"])
log_path = Path(os.environ["EVAL_VLLM_LOG"])
timeout = int(os.environ["EVAL_VLLM_STARTUP_TIMEOUT"])
api_key = os.environ.get("VLLM_API_KEY", "")
url = f"http://{host}:{port}/v1/models"
deadline = time.time() + timeout
last_status = 0.0

def proc_is_alive() -> bool:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            parts = proc_stat.read_text().split()
            if len(parts) >= 3 and parts[2] == "Z":
                return False
        except OSError:
            pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

def tail_log(num_lines: int = 120) -> str:
    if not log_path.exists():
        return f"{log_path} does not exist yet"
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"Could not read {log_path}: {exc}"
    return "\\n".join(lines[-num_lines:]) if lines else f"{log_path} is empty"

while time.time() < deadline:
    if not proc_is_alive():
        raise SystemExit(f"vLLM process {pid} exited before becoming ready. Last log lines:\\n{tail_log()}")
    try:
        request = urllib.request.Request(url)
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status == 200:
                print(f"vLLM is ready at {url}")
                raise SystemExit(0)
    except (urllib.error.URLError, TimeoutError):
        now = time.time()
        if now - last_status >= 60:
            remaining = int(deadline - now)
            print(f"Still waiting for vLLM at {url}; {remaining}s before timeout; log={log_path}", flush=True)
            last_status = now
        time.sleep(5)
raise SystemExit(f"Timed out after {timeout}s waiting for vLLM at {url}. Last log lines:\\n{tail_log()}")
PY

RUN_ARGS=(
  --container-runtime "${CONTAINER_RUNTIME}"
  --comp "${EVAL_COMP}"
  --models "${EVAL_MODEL_CONFIG}"
  --n "${EVAL_N}"
  --output-dir "${EVAL_OUTPUT_DIR}"
)

if [[ "${EVAL_REDO_ALL:-false}" == "true" ]]; then
  RUN_ARGS+=(--redo-all)
fi

if [[ -n "${EVAL_PROBLEMS:-}" && "${EVAL_NUM_SHARDS}" -gt 1 ]]; then
  echo "Set either EVAL_PROBLEMS or EVAL_NUM_SHARDS, not both." >&2
  exit 1
fi

if [[ -n "${EVAL_PROBLEMS:-}" ]]; then
  # shellcheck disable=SC2206
  PROBLEM_ARGS=(${EVAL_PROBLEMS})
  RUN_ARGS+=(--problems "${PROBLEM_ARGS[@]}")
elif (( EVAL_NUM_SHARDS > 1 )); then
  mapfile -t PROBLEM_ARGS < <(
    python "${REPO_ROOT}/train/scripts/eval_problem_shard.py" \
      --comp "${EVAL_COMP}" \
      --shard-index "${EVAL_SHARD_INDEX}" \
      --num-shards "${EVAL_NUM_SHARDS}" \
      --configs-dir "${REPO_ROOT}/configs/competitions"
  )
  if (( ${#PROBLEM_ARGS[@]} == 0 )); then
    echo "Shard ${EVAL_SHARD_INDEX}/${EVAL_NUM_SHARDS} has no problems for ${EVAL_COMP}." >&2
    exit 1
  fi
  last_problem_idx=$((${#PROBLEM_ARGS[@]} - 1))
  echo "Eval shard ${EVAL_SHARD_INDEX}/${EVAL_NUM_SHARDS}: ${#PROBLEM_ARGS[@]} problems (${PROBLEM_ARGS[0]}..${PROBLEM_ARGS[$last_problem_idx]})"
  RUN_ARGS+=(--problems "${PROBLEM_ARGS[@]}")
fi

cd "${REPO_ROOT}"
python scripts/run.py "${RUN_ARGS[@]}"
