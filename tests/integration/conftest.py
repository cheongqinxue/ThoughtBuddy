"""Shared fixtures for live-service integration tests."""

from __future__ import annotations

import pytest

from integration.support import require_vllm


@pytest.fixture
def live_vllm_base_url() -> str:
    """Fixture that skips when local vLLM is not running."""
    return require_vllm()
