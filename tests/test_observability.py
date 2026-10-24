"""Tests for observability helpers (v1 Langfuse backend)."""

from __future__ import annotations

import importlib
import logging
from unittest.mock import MagicMock, patch

import pytest

from config import Settings
from utils.observability import (
    LangfuseLogHandler,
    LangfuseWakeTrace,
    apply_studio_langfuse_config,
    delete_langfuse_trace,
    flush_langfuse,
    is_langfuse_configured,
    is_langfuse_enabled,
    langfuse_direct_trace,
    langfuse_wake_trace,
    normalize_trace_id,
)


class TestLangfuseAvailability:
    """Langfuse enablement checks."""

    def test_is_langfuse_configured_requires_both_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Require public and secret keys."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        assert is_langfuse_configured() is False

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        assert is_langfuse_configured() is True

    def test_is_langfuse_enabled_requires_config_and_keys(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Respect config flag and credential presence."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

        disabled = Settings(langfuse_enabled=False)
        enabled = Settings(langfuse_enabled=True)

        assert is_langfuse_enabled(disabled) is False
        assert is_langfuse_enabled(enabled) is True


class TestTraceHelpers:
    """Trace id and invoke config helpers."""

    def test_normalize_trace_id_strips_uuid_dashes(self) -> None:
        """Convert UUID wake ids to Langfuse-friendly hex ids."""
        wake_id = "550e8400-e29b-41d4-a716-446655440000"
        assert normalize_trace_id(wake_id) == "550e8400e29b41d4a716446655440000"

    def test_invoke_config_includes_callbacks_and_metadata(self) -> None:
        """Build graph config with Langfuse callbacks."""
        handler = MagicMock()
        trace = LangfuseWakeTrace(
            handler=handler,
            callbacks=[handler],
            run_name="thoughtbuddy-wake",
            tags=("thoughtbuddy", "wake"),
        )

        config = trace.invoke_config("thoughtbuddy-main", extra_metadata={"source": "wake"})

        assert config["callbacks"] == [handler]
        assert config["run_name"] == "thoughtbuddy-wake"
        assert config["configurable"]["thread_id"] == "thoughtbuddy-main"
        assert config["metadata"]["langfuse_session_id"] == "thoughtbuddy-main"
        assert config["metadata"]["source"] == "wake"
        assert "thoughtbuddy" in config["metadata"]["langfuse_tags"]


class TestLangfuseWakeTrace:
    """Wake trace context manager tests."""

    def test_yields_none_when_disabled(self) -> None:
        """Skip tracing when Langfuse is disabled."""
        settings = Settings(langfuse_enabled=False)

        with langfuse_wake_trace(
            settings,
            wake_id="wake-1",
            thread_id="thread-1",
            task_ids=["task-1"],
        ) as trace:
            assert trace is None

    def test_creates_handler_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Create a callback handler and propagate wake metadata."""
        importlib.import_module("langfuse.langchain")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        mock_handler = MagicMock()
        mock_propagate = MagicMock()
        mock_propagate.return_value.__enter__ = MagicMock(return_value=None)
        mock_propagate.return_value.__exit__ = MagicMock(return_value=False)

        settings = Settings(langfuse_enabled=True)

        with (
            patch("langfuse.langchain.CallbackHandler", return_value=mock_handler) as mock_handler_cls,
            patch("langfuse.propagate_attributes", mock_propagate),
        ):
            with langfuse_wake_trace(
                settings,
                wake_id="550e8400-e29b-41d4-a716-446655440000",
                thread_id="thoughtbuddy-main",
                task_ids=["task-1"],
            ) as trace:
                assert trace is not None
                assert trace.callbacks == [mock_handler]

            mock_handler_cls.assert_called_once_with(
                trace_context={"trace_id": "550e8400e29b41d4a716446655440000"},
            )
            mock_propagate.assert_called_once()


class TestLangfuseLogHandler:
    """Log bridge tests."""

    def test_forwards_warning_records_to_current_span(self) -> None:
        """Attach WARNING logs to the active Langfuse span."""
        handler = LangfuseLogHandler(min_level=logging.WARNING)
        record = logging.LogRecord(
            name="thoughtbuddy.tools.websearch",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="tool_failed_recoverable",
            args=(),
            exc_info=None,
        )
        record.wake_id = "wake-1"
        record.thread_id = "thread-1"

        mock_client = MagicMock()
        with patch("langfuse.get_client", return_value=mock_client):
            handler.emit(record)

        mock_client.update_current_span.assert_called_once()
        kwargs = mock_client.update_current_span.call_args.kwargs
        assert kwargs["level"] == "ERROR"
        assert "tool_failed_recoverable" in kwargs["status_message"]


class TestLangfuseDirectTrace:
    """Direct invoke trace tests."""

    def test_yields_none_when_disabled(self) -> None:
        """Skip tracing when Langfuse is disabled."""
        settings = Settings(langfuse_enabled=False)

        with langfuse_direct_trace(settings, thread_id="thread-1") as trace:
            assert trace is None


class TestStudioLangfuseConfig:
    """LangGraph Studio graph wrapper tests."""

    def test_apply_studio_langfuse_config_noop_when_disabled(self) -> None:
        """Return the original graph when Langfuse is disabled."""
        graph = MagicMock()

        result = apply_studio_langfuse_config(graph, Settings(langfuse_enabled=False))

        assert result is graph
        graph.with_config.assert_not_called()

    def test_apply_studio_langfuse_config_wraps_graph_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Attach default callbacks for Studio invocations."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        graph = MagicMock()
        wrapped = MagicMock()
        graph.with_config.return_value = wrapped
        callback = MagicMock()

        with patch(
            "utils.observability._studio_callback_handler",
            return_value=callback,
        ):
            result = apply_studio_langfuse_config(graph, Settings(langfuse_enabled=True))

        assert result is wrapped
        graph.with_config.assert_called_once_with(
            {
                "callbacks": [callback],
                "run_name": "thoughtbuddy-studio",
                "metadata": {
                    "langfuse_tags": ["thoughtbuddy", "studio"],
                },
            },
        )


class TestLangfuseFlush:
    """Flush helper tests."""

    def test_flush_noop_when_disabled(self) -> None:
        """Do nothing when Langfuse is disabled."""
        flush_langfuse(Settings(langfuse_enabled=False))


class TestLangfuseDelete:
    """Trace deletion helper tests."""

    def test_delete_noop_when_disabled(self) -> None:
        """Skip delete API calls when Langfuse is disabled."""
        assert delete_langfuse_trace(Settings(langfuse_enabled=False), "trace-1") is False

    def test_delete_calls_langfuse_api_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Delete a trace via the Langfuse API client."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        mock_client = MagicMock()

        settings = Settings(langfuse_enabled=True, langfuse_base_url="http://localhost:3001")

        with patch("langfuse.get_client", return_value=mock_client):
            assert delete_langfuse_trace(settings, "550e8400-e29b-41d4-a716-446655440000") is True

        mock_client.api.trace.delete.assert_called_once_with("550e8400e29b41d4a716446655440000")
