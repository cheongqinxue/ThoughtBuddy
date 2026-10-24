#! /bin/bash
# Start local vLLM for the ThoughtBuddy ReAct harness (tool calling required).
#
# Profiles (set VLLM_PROFILE or override individual vars):
#   qwen3  Qwen/Qwen3-8B-AWQ + hermes + awq (default)
#   lfm    LiquidAI/LFM2.5-8B-A1B + lfm2 + qwen3 reasoning parser
#
# Usage:
#   ./scripts/start_vllm_agent.sh
#   VLLM_PROFILE=lfm ./scripts/start_vllm_agent.sh
#   ./scripts/set_llm_profile.sh lfm   # align config.yaml, then restart stack

set -e

VLLM="${HOME}/anaconda3/envs/llm/bin/vllm"
PYTHON="${HOME}/anaconda3/envs/llm/bin/python"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_vllm_profiles.sh"
apply_vllm_profile "${VLLM_PROFILE:-qwen3}"

MODEL="${VLLM_MODEL}"
TOOL_PARSER="${VLLM_TOOL_PARSER}"

# shellcheck source=/dev/null
source "$(dirname "$0")/_vllm_cuda_env.sh"

# Prefer HF cache on external drive when mounted; fall back to ~/.cache (common when drive is away).
_local_hf="${HOME}/.cache/huggingface"
_requested_hf="${HF_HOME:-}"
if [ -n "${_requested_hf}" ] && mkdir -p "${_requested_hf}" 2>/dev/null && touch "${_requested_hf}/.write_test" 2>/dev/null; then
  rm -f "${_requested_hf}/.write_test"
  export HF_HOME="${_requested_hf}"
else
  export HF_HOME="${_local_hf}"
  echo "HF cache not writable; using ${HF_HOME}"
fi
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "${HUGGINGFACE_HUB_CACHE}"

ARGS=(
  serve "${MODEL}"
  --enable-auto-tool-choice
  --tool-call-parser "${TOOL_PARSER}"
)

if [ -n "${VLLM_QUANTIZATION:-}" ]; then
  ARGS+=(--quantization "${VLLM_QUANTIZATION}")
elif [[ "${MODEL}" == *AWQ* ]]; then
  ARGS+=(--quantization awq)
fi

if [ "${#VLLM_PROFILE_EXTRA_ARGS[@]}" -gt 0 ]; then
  ARGS+=("${VLLM_PROFILE_EXTRA_ARGS[@]}")
fi

if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA=( ${VLLM_EXTRA_ARGS} )
  ARGS+=("${EXTRA[@]}")
fi

echo "Starting vLLM: profile=${VLLM_PROFILE:-qwen3} model=${MODEL} tool_call_parser=${TOOL_PARSER}"
echo "HF_HOME=${HF_HOME}"
exec env \
  HF_HOME="${HF_HOME}" \
  HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
  HF_HUB_CACHE="${HF_HUB_CACHE}" \
  TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  "${VLLM}" "${ARGS[@]}"
