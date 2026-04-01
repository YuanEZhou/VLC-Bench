#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"


IMAGE="${IMAGE:-zhouyuanen/llama.cpp:server}"
CONTAINER_NAME="${CONTAINER_NAME:-llamacpp-cpu}"
HOST_PORT="${HOST_PORT:-8010}"
CONTAINER_PORT="${CONTAINER_PORT:-8080}"
MODEL_GGUF="${MODEL_GGUF:-$PWD/Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf}"
MODELS_DIR="${MODELS_DIR:-}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$PWD/hf-cache}"
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-}"
BUILTIN_TOOLS="${BUILTIN_TOOLS:-}"

mkdir -p "${HF_CACHE_DIR}"

MODEL_ARGS=()
MOUNT_ARGS=( -v "${HF_CACHE_DIR}:/root/.cache/huggingface" )
TOOL_ARGS=()

if [[ ! -f "${MODEL_GGUF}" ]]; then
  echo "ERROR: MODEL_GGUF does not exist: ${MODEL_GGUF}" >&2
  exit 1
fi

MODEL_GGUF_ABS="$(readlink -f "${MODEL_GGUF}")"
if [[ -z "${MODELS_DIR}" ]]; then
  MODELS_DIR="$(dirname "${MODEL_GGUF_ABS}")"
fi
MODELS_DIR_ABS="$(readlink -f "${MODELS_DIR}")"

if [[ ! -d "${MODELS_DIR_ABS}" ]]; then
  echo "ERROR: MODELS_DIR does not exist: ${MODELS_DIR_ABS}" >&2
  exit 1
fi

MODEL_GGUF_BASE="$(basename "${MODEL_GGUF_ABS}")"
MOUNT_ARGS+=( -v "${MODELS_DIR_ABS}:/models:ro" )
MODEL_ARGS=( -m "/models/${MODEL_GGUF_BASE}" )

if [[ "${ENABLE_TOOL_CALLING}" == "1" ]]; then
  # Force jinja template parsing so tool calls can be emitted as structured fields.
  TOOL_ARGS+=( --jinja --no-skip-chat-parsing )
  if [[ -n "${CHAT_TEMPLATE}" ]]; then
    TOOL_ARGS+=( --chat-template "${CHAT_TEMPLATE}" )
  fi
  if [[ -n "${CHAT_TEMPLATE_FILE}" ]]; then
    TOOL_ARGS+=( --chat-template-file "${CHAT_TEMPLATE_FILE}" )
  fi
fi

if [[ -n "${BUILTIN_TOOLS}" ]]; then
  TOOL_ARGS+=( --tools "${BUILTIN_TOOLS}" )
fi

echo "Removing old container: ${CONTAINER_NAME}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Starting llama.cpp (CPU), port ${HOST_PORT} -> ${CONTAINER_PORT}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --cpus "4" \
  --memory "4g" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -e "HF_ENDPOINT=${HF_ENDPOINT}" \
  "${MOUNT_ARGS[@]}" \
  "${IMAGE}" \
  "${MODEL_ARGS[@]}" \
  --threads 4 \
  --parallel 4 \
  --ctx-size 4096 \
  --host 0.0.0.0 \
  --port "${CONTAINER_PORT}" \
  "${TOOL_ARGS[@]}"

echo "Done."
echo "Logs: docker logs -f ${CONTAINER_NAME}"
echo "Stop: docker rm -f ${CONTAINER_NAME}"
