# vLLM agent profiles for ThoughtBuddy (source from start_vllm_agent.sh).
#
# Usage:
#   VLLM_PROFILE=qwen3 ./scripts/start_vllm_agent.sh   # default
#   VLLM_PROFILE=lfm ./scripts/start_vllm_agent.sh
#
# Align ThoughtBuddy config with: ./scripts/set_llm_profile.sh <profile>

apply_vllm_profile() {
  local profile="${1:-qwen3}"
  VLLM_PROFILE_EXTRA_ARGS=()

  case "${profile}" in
    qwen3)
      export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B-AWQ}"
      export VLLM_TOOL_PARSER="${VLLM_TOOL_PARSER:-hermes}"
      export VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-awq}"
      ;;
    lfm)
      export VLLM_MODEL="${VLLM_MODEL:-LiquidAI/LFM2.5-8B-A1B}"
      export VLLM_TOOL_PARSER="${VLLM_TOOL_PARSER:-lfm2}"
      unset VLLM_QUANTIZATION
      VLLM_PROFILE_EXTRA_ARGS=(--reasoning-parser qwen3)
      ;;
    *)
      echo "error: unknown VLLM_PROFILE=${profile} (expected qwen3 or lfm)" >&2
      return 1
      ;;
  esac
}
