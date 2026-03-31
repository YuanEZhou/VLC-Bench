#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-1.7b-cpu}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-cpu-qwen}"
HOST_PORT="${HOST_PORT:-8001}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$PWD/hf-cache}"
IMAGE="${IMAGE:-vllm/vllm-openai-cpu:latest-x86_64}"
MODEL_LOCAL_DIR="${MODEL_LOCAL_DIR:-}"

mkdir -p "${HF_CACHE_DIR}"

MODEL_ARG="${MODEL_NAME}"
MODEL_MOUNT_ARGS=()
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

echo "Starting vLLM on CPU, port ${HOST_PORT}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --ipc=host \
  -e "HF_ENDPOINT=${HF_ENDPOINT}" \
  -p "${HOST_PORT}:8000" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "${MODEL_MOUNT_ARGS[@]}" \
  "${IMAGE}" \
  "${MODEL_ARG}" \
  --served-model-name "${SERVED_MODEL_NAME}"

echo "Done."
echo "Logs: docker logs -f ${CONTAINER_NAME}"
echo "Stop: docker rm -f ${CONTAINER_NAME}"
