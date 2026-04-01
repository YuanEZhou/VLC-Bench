#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"


MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-1.7b}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-gpu-qwen}"
GPU_ID="${GPU_ID:-0}"
HOST_PORT="${HOST_PORT:-8000}"
THREADS="${THREADS:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$PWD/hf-cache}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
MODEL_LOCAL_DIR="${MODEL_LOCAL_DIR:-}"

mkdir -p "${HF_CACHE_DIR}"

MODEL_ARG="${MODEL_NAME}"
MODEL_MOUNT_ARGS=()

# If local model directory is provided, mount it and pass container path to --model.
if [[ -n "${MODEL_LOCAL_DIR}" ]]; then
  if [[ ! -d "${MODEL_LOCAL_DIR}" ]]; then
    echo "ERROR: MODEL_LOCAL_DIR does not exist: ${MODEL_LOCAL_DIR}" >&2
    exit 1
  fi
  MODEL_MOUNT_ARGS=( -v "${MODEL_LOCAL_DIR}:/models/local-model:ro" )
  MODEL_ARG="/models/local-model"
fi

echo "Removing old container: ${CONTAINER_NAME}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Starting vLLM on GPU ${GPU_ID}, port ${HOST_PORT}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --cpus "4" \
  --memory "4g" \
  --gpus "device=${GPU_ID}" \
  --ipc=host \
  -e "OMP_NUM_THREADS=${THREADS}" \
  -e "MKL_NUM_THREADS=${THREADS}" \
  -e "HF_ENDPOINT=${HF_ENDPOINT}" \
  -p "${HOST_PORT}:8000" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "${MODEL_MOUNT_ARGS[@]}" \
  "${IMAGE}" \
  --model "${MODEL_ARG}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"

echo "Done."
echo "Logs: docker logs -f ${CONTAINER_NAME}"
echo "Stop: docker rm -f ${CONTAINER_NAME}"
