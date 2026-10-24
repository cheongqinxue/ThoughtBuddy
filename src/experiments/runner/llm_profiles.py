"""LLM profile definitions for experiment stack runs."""

from __future__ import annotations

from typing import Any, TypedDict


class LlmProfileValues(TypedDict):
    """Fields synced into config.yaml under llm:."""

    profile: str
    agent_model: str
    enable_thinking: bool
    context_window_tokens: int


LLM_PROFILES: dict[str, LlmProfileValues] = {
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


def resolve_llm_profile(profile: str) -> LlmProfileValues:
    """Return LLM settings for a named profile.

    Args:
        profile: Profile id (``qwen3`` or ``lfm``).

    Returns:
        Values to write under ``llm:`` in config.yaml.

    Raises:
        ValueError: When the profile name is unknown.
    """
    if profile not in LLM_PROFILES:
        msg = f"unknown llm profile {profile!r} (expected qwen3 or lfm)"
        raise ValueError(msg)
    return LLM_PROFILES[profile]


def llm_profile_as_yaml_values(profile: str) -> dict[str, Any]:
    """Return llm section fields for merging into config.yaml.

    Args:
        profile: Profile id.

    Returns:
        Flat dict of llm keys suitable for YAML merge.
    """
    return dict(resolve_llm_profile(profile))
