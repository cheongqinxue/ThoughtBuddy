"""Token budget and compaction threshold helpers."""

from __future__ import annotations

from config import Settings
from llm.profile import resolve_model_profile

_DEFAULT_COMPACT_RATIO = 0.9


def max_prompt_tokens(settings: Settings) -> int:
    """Return the maximum allowed estimated prompt size for one LLM call.

    Args:
        settings: Application settings.

    Returns:
        Prompt token budget derived from context window minus completion reserve.
    """
    return resolve_model_profile(settings).max_prompt_tokens


def compaction_enabled(_settings: Settings) -> bool:
    """Return whether archive compaction is active.

    Compaction is always enabled in v1. Threshold tuning uses
    ``context.compact_token_threshold`` or ``context.compact_token_ratio``.

    Args:
        _settings: Application settings (unused; kept for call-site stability).

    Returns:
        Always ``True``.
    """
    return True


def effective_compact_ratio(settings: Settings) -> float:
    """Return the ratio used when no explicit compaction threshold is set.

    Args:
        settings: Application settings.

    Returns:
        Configured ratio, or ``0.9`` when unset or non-positive.
    """
    ratio = settings.context_compact_token_ratio
    if ratio <= 0:
        return _DEFAULT_COMPACT_RATIO
    return ratio


def compaction_token_threshold(settings: Settings) -> int:
    """Return the token count that triggers archive compaction.

    Uses ``context.compact_token_threshold`` when set; otherwise
    ``effective_compact_ratio()`` × ``max_prompt_tokens`` (default 90%).

    Args:
        settings: Application settings.

    Returns:
        Estimated token count above which compaction should run.
    """
    explicit = settings.context_compact_token_threshold
    if explicit > 0:
        return explicit
    return int(max_prompt_tokens(settings) * effective_compact_ratio(settings))
