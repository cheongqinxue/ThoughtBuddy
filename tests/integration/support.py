"""Helpers for live-service integration tests."""

from __future__ import annotations

import os
import time

import httpx
import pytest

from config import Settings


def wait_for_http_ok(url: str, *, timeout_seconds: int = 180, interval_seconds: float = 2.0) -> None:
    """Poll an HTTP URL until it returns 200.

    Args:
        url: URL to poll.
        timeout_seconds: Maximum seconds to wait.
        interval_seconds: Seconds between attempts.

    Raises:
        RuntimeError: If the URL does not become healthy in time.
    """
    deadline = time.time() + timeout_seconds
    last_error = "unknown"

    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(interval_seconds)

    msg = f"Service not healthy at {url} after {timeout_seconds}s: {last_error}"
    raise RuntimeError(msg)


def vllm_base_url() -> str:
    """Return the OpenAI-compatible vLLM base URL."""
    return os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")


def vllm_model_name() -> str:
    """Return the model name configured for the agent harness."""
    override = os.getenv("VLLM_MODEL", "").strip()
    if override:
        return override
    return Settings.load().agent_model


def require_vllm(base_url: str | None = None, *, smoke: bool | None = None) -> str:
    """Ensure vLLM responds before running a live inference test.

    Args:
        base_url: Optional override for the vLLM base URL.
        smoke: When True, fail instead of skip if vLLM is down. Defaults to
            ``THOUGHTBUDDY_VLLM_SMOKE`` env.

    Returns:
        Resolved vLLM base URL.

    Raises:
        pytest.skip: vLLM is unavailable and smoke mode is off.
        pytest.fail: vLLM is unavailable and smoke mode is on.
    """
    resolved = (base_url or vllm_base_url()).rstrip("/")
    smoke_mode = smoke if smoke is not None else bool(os.getenv("THOUGHTBUDDY_VLLM_SMOKE"))

    try:
        wait_for_http_ok(f"{resolved}/models", timeout_seconds=5, interval_seconds=1.0)
    except RuntimeError as exc:
        if smoke_mode:
            pytest.fail(str(exc))
        pytest.skip(str(exc))

    return resolved
