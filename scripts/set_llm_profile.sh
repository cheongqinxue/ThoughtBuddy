#!/usr/bin/env bash
# Switch config.yaml llm settings to match a vLLM agent profile.
#
# For chess experiment runs, prefer config/experiments/*.yaml + run.sh start
# (applies llm profile and starts vLLM with matching VLLM_PROFILE).
#
# Usage (manual dev without run.sh):
#   ./scripts/set_llm_profile.sh qwen3
#   ./scripts/set_llm_profile.sh lfm
#
# Then restart vLLM and langgraph dev:
#   VLLM_PROFILE=<profile> ./scripts/start_vllm_agent.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-}"

if [[ -z "${PROFILE}" ]]; then
  cat >&2 <<EOF
usage: $(basename "$0") <qwen3|lfm>

Profiles:
  qwen3  Qwen/Qwen3-8B-AWQ, hermes parser, 32K context (default)
  lfm    LiquidAI/LFM2.5-8B-A1B, lfm2 parser, 128K context, always-on reasoning
EOF
  exit 1
fi

cd "${ROOT}"
uv run python - "${PROFILE}" <<'PY'
import re
import sys
from pathlib import Path

PROFILE = sys.argv[1]
CONFIG_PATH = Path("config.yaml")

PROFILES = {
    "qwen3": {
        "profile": "qwen3",
        "agent_model": "Qwen/Qwen3-8B-AWQ",
        "enable_thinking": False,
        "context_window_tokens": 32768,
    },
    "lfm": {
        "profile": "lfm",
        "agent_model": "LiquidAI/LFM2.5-8B-A1B",
        "enable_thinking": False,
        "context_window_tokens": 131072,
    },
}

if PROFILE not in PROFILES:
    raise SystemExit(f"unknown profile: {PROFILE!r} (expected qwen3 or lfm)")

chosen = PROFILES[PROFILE]
text = CONFIG_PATH.read_text(encoding="utf-8")

for key, value in chosen.items():
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = str(value)
    pattern = rf"^(  {re.escape(key)}:\s*)(.+)$"
    new_text, count = re.subn(pattern, rf"\g<1>{rendered}", text, count=1, flags=re.MULTILINE)
    if count == 0:
        raise SystemExit(f"config.yaml missing llm.{key} (expected under llm:)")
    text = new_text

CONFIG_PATH.write_text(text, encoding="utf-8")

print(f"Updated config.yaml llm.profile={chosen['profile']}")
print(f"  agent_model: {chosen['agent_model']}")
print(f"  context_window_tokens: {chosen['context_window_tokens']}")
print()
print(f"Restart stack with: VLLM_PROFILE={PROFILE} ./scripts/start_vllm_agent.sh")
PY
