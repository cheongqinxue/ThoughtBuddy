"""Live Langfuse integration tests (require running Langfuse + credentials)."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from dotenv import load_dotenv
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from config import Settings
from db.connection import connect
from graph.builder import build_graph
from runtime.scratchpad_layout import ensure_scratchpad_layout
from runtime.wake import build_wake_context, invoke_wake_cycle
from store.tasks import Task
from utils.observability import (
    configure_langfuse_environment,
    delete_langfuse_trace,
    flush_langfuse,
    is_langfuse_configured,
    is_langfuse_enabled,
    langfuse_direct_trace,
    normalize_trace_id,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SMOKE_THREAD_ID = "langfuse-smoke-test"


def _integration_settings() -> Settings:
    """Build settings for live Langfuse integration tests."""
    return Settings(
        openai_base_url="http://localhost:8000/v1",
        openai_api_key="test-key",
        agent_model="test-model",
        langsearch_api_key="test-key",
        langfuse_enabled=True,
        langfuse_base_url="http://localhost:3001",
    )


def _wait_for_langfuse_health(base_url: str, *, timeout_seconds: int = 180) -> None:
    """Wait until Langfuse responds on its public health endpoint.

    Args:
        base_url: Langfuse base URL.
        timeout_seconds: Maximum seconds to wait.

    Raises:
        RuntimeError: If Langfuse does not become healthy in time.
    """
    health_url = f"{base_url.rstrip('/')}/api/public/health"
    deadline = time.time() + timeout_seconds
    last_error = "unknown"

    while time.time() < deadline:
        try:
            response = httpx.get(health_url, timeout=5.0)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(2)

    msg = f"Langfuse not healthy at {health_url} after {timeout_seconds}s: {last_error}"
    raise RuntimeError(msg)


def _require_langfuse_configured() -> None:
    """Fail fast during smoke runs when credentials were expected to be synced."""
    if is_langfuse_configured():
        return
    if os.getenv("THOUGHTBUDDY_LANGFUSE_SMOKE"):
        pytest.fail("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env")
    pytest.skip("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")


def _langfuse_auth() -> tuple[str, str]:
    """Return Langfuse basic-auth credentials from the environment."""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        msg = "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
        raise RuntimeError(msg)
    return public_key, secret_key


def _wait_for_trace_observation(
    settings: Settings,
    trace_id: str,
    *,
    timeout_seconds: int = 30,
) -> None:
    """Poll Langfuse until an observation exists for the given trace id.

    Args:
        settings: Application settings.
        trace_id: Trace identifier to wait for.
        timeout_seconds: Maximum seconds to wait.

    Raises:
        RuntimeError: If the trace never appears.
    """
    auth = _langfuse_auth()
    base_url = settings.langfuse_base_url.rstrip("/")
    normalized_trace_id = normalize_trace_id(trace_id)
    deadline = time.time() + timeout_seconds
    last_error = "observation not found"

    while time.time() < deadline:
        try:
            response = httpx.get(
                f"{base_url}/api/public/v2/observations",
                params={"limit": 50},
                auth=auth,
                timeout=10.0,
            )
            response.raise_for_status()
            observations = response.json().get("data") or []
            if any(item.get("traceId") == normalized_trace_id for item in observations):
                return
            last_error = f"trace {normalized_trace_id} not in latest observations"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)

    msg = f"Expected Langfuse observations for trace {normalized_trace_id}: {last_error}"
    raise RuntimeError(msg)


def _cleanup_langfuse_trace(settings: Settings, trace_id: str) -> None:
    """Queue Langfuse trace deletion after an integration test.

    Langfuse deletes tracing data asynchronously (often within ~15 minutes).
    The smoke tests verify trace creation, then queue cleanup so the UI is not
    permanently cluttered with test data.

    Args:
        settings: Application settings.
        trace_id: Trace identifier to delete.
    """
    deleted = delete_langfuse_trace(settings, trace_id)
    if not deleted:
        pytest.fail(f"Failed to queue Langfuse deletion for trace {trace_id}")


class TestLangfuseLiveIntegration:
    """End-to-end checks against a running Langfuse instance."""

    def test_langfuse_health_endpoint(self) -> None:
        """Langfuse web health endpoint should respond when the stack is up."""
        settings = _integration_settings()
        try:
            _wait_for_langfuse_health(settings.langfuse_base_url, timeout_seconds=5)
        except RuntimeError as exc:
            pytest.skip(str(exc))

    def test_direct_model_invoke_creates_queryable_trace(self) -> None:
        """A traced LangChain invoke should appear in Langfuse and be cleaned up."""
        _require_langfuse_configured()

        settings = _integration_settings()
        if not is_langfuse_enabled(settings):
            pytest.skip("Enable observability.langfuse.enabled and configure credentials")

        configure_langfuse_environment(settings)
        _wait_for_langfuse_health(settings.langfuse_base_url)

        invoke_id = str(uuid.uuid4())
        model = FakeListChatModel(responses=["langfuse integration ok"])
        trace_id = invoke_id

        try:
            with langfuse_direct_trace(
                settings,
                thread_id="integration-test",
                invoke_id=invoke_id,
            ) as langfuse_trace:
                assert langfuse_trace is not None
                result = model.invoke(
                    [HumanMessage(content="langfuse integration ping")],
                    config=langfuse_trace.invoke_config("integration-test"),
                )
                assert "langfuse integration ok" in str(result.content)

            flush_langfuse(settings)

            last_trace_id = langfuse_trace.handler.last_trace_id
            assert last_trace_id, "Langfuse callback handler did not record a trace id"
            trace_id = last_trace_id
            _wait_for_trace_observation(settings, trace_id)
        finally:
            _cleanup_langfuse_trace(settings, trace_id)

    def test_wake_invoke_creates_queryable_trace_and_cleans_up(
        self,
        tmp_path: Path,
    ) -> None:
        """A traced queue wake should appear in Langfuse and be deleted afterward."""
        _require_langfuse_configured()

        loaded = Settings.load()
        if not loaded.langfuse_enabled:
            pytest.skip("Enable observability.langfuse.enabled in config.yaml")

        settings = loaded.model_copy(
            update={
                "openai_api_key": "test-key",
                "langsearch_api_key": "test-key",
                "langfuse_base_url": loaded.langfuse_base_url or "http://localhost:3001",
                "thoughtbuddy_db_path": str(tmp_path / "smoke.db"),
                "scratchpad_directory": str(tmp_path / "scratchpad"),
                "context_compact_token_threshold": 999_999_999,
            },
        )
        if not is_langfuse_enabled(settings):
            pytest.skip("Configure LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")

        configure_langfuse_environment(settings)
        _wait_for_langfuse_health(settings.langfuse_base_url)

        scratchpad_dir = tmp_path / "scratchpad"
        ensure_scratchpad_layout(scratchpad_dir, template_dir=Path(".data-template/scratchpad"))
        conn = connect(tmp_path / "smoke.db")

        pending = [
            Task(
                id="smoke-task-1",
                alert_id=None,
                occurrence_at=None,
                source="user",
                status="pending",
                payload={"message": "langfuse smoke wake"},
                disposition_reason=None,
                created_at="2026-09-13 08:00:00",
                updated_at="2026-09-13 08:00:00",
                started_at=None,
                completed_at=None,
            ),
        ]
        context = build_wake_context(
            pending_tasks=pending,
            upcoming_alerts=[],
            queue_stats={"pending": 1},
        )

        wake_id = str(uuid.uuid4())
        model = FakeListChatModel(responses=["phase complete"] * 12)
        graph = build_graph(model, tools=[])

        try:
            with patch(
                "runtime.wake.uuid.uuid4",
                return_value=uuid.UUID(wake_id),
            ):
                invoke_wake_cycle(
                    graph,
                    context,
                    conn,
                    settings,
                    model,
                    thread_id=SMOKE_THREAD_ID,
                )

            _wait_for_trace_observation(settings, wake_id)
        finally:
            _cleanup_langfuse_trace(settings, wake_id)
