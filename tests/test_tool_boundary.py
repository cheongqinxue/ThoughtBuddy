"""Tests for shared agent tool boundary helpers."""

from __future__ import annotations

import json
import logging

import pytest
from langchain_core.tools import StructuredTool

from utils.api_retry import RetryableApiError
from tools.tool_boundary import (
    is_tool_boundary_wrapped,
    run_tool_boundary,
    tool_error_payload,
    wrap_tool_with_boundary,
)


class TestToolErrorPayload:
    """Structured tool error payload tests."""

    def test_payload_includes_retryable_flag(self) -> None:
        """Include retry metadata for the agent."""
        payload = json.loads(
            tool_error_payload("websearch", RetryableApiError("offline"), retryable=True),
        )

        assert payload["ok"] is False
        assert payload["tool"] == "websearch"
        assert payload["retryable"] is True
        assert "offline" in payload["message"]


class TestRunToolBoundary:
    """Tool boundary execution tests."""

    def test_returns_result_on_success(self) -> None:
        """Return successful tool output unchanged."""
        result = run_tool_boundary("demo", lambda value: f"ok:{value}", "x")

        assert result == "ok:x"

    def test_returns_json_and_logs_recoverable_error(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Log recoverable failures with traceback and return JSON to the agent."""

        def fail() -> str:
            raise RetryableApiError("network down")

        with caplog.at_level(logging.ERROR, logger="thoughtbuddy.tools.demo"):
            result = run_tool_boundary("demo", fail)

        payload = json.loads(result)
        assert payload["ok"] is False
        assert payload["retryable"] is True
        assert any(record.levelname == "ERROR" for record in caplog.records)

    def test_reraises_unexpected_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        """Propagate unexpected failures after logging them."""

        def fail() -> None:
            msg = "logic broke"
            raise RuntimeError(msg)

        with caplog.at_level(logging.ERROR, logger="thoughtbuddy.tools.demo"):
            with pytest.raises(RuntimeError, match="logic broke"):
                run_tool_boundary("demo", fail)

        assert any(record.levelname == "ERROR" for record in caplog.records)


class TestWrapToolWithBoundary:
    """Central tool wrapping tests."""

    def test_wraps_structured_tool_func(self) -> None:
        """Wrap func and mark tool as boundary-protected."""

        def demo(value: str) -> str:
            return f"ok:{value}"

        tool = StructuredTool.from_function(
            func=demo,
            name="demo",
            description="demo tool",
        )
        wrapped = wrap_tool_with_boundary(tool)

        assert wrapped is not tool
        assert is_tool_boundary_wrapped(wrapped)
        assert wrapped.invoke({"value": "x"}) == "ok:x"

    def test_wrap_is_idempotent(self) -> None:
        """Applying the wrapper twice should not nest boundaries."""

        tool = StructuredTool.from_function(
            func=lambda: "ok",
            name="demo",
            description="demo tool",
        )
        once = wrap_tool_with_boundary(tool)
        twice = wrap_tool_with_boundary(once)

        assert twice is once

    def test_value_error_returns_json_payload(self) -> None:
        """Agent input errors should return structured JSON to the model."""

        def fail() -> str:
            msg = "run_at is required"
            raise ValueError(msg)

        tool = wrap_tool_with_boundary(
            StructuredTool.from_function(
                func=fail,
                name="create_alert",
                description="demo",
            ),
        )
        payload = json.loads(tool.invoke({}))

        assert payload["ok"] is False
        assert payload["tool"] == "create_alert"
        assert payload["error_type"] == "ValueError"
        assert "run_at" in payload["message"]
