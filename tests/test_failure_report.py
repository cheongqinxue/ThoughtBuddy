"""Tests for experiment failure reports."""

from __future__ import annotations

import json
from pathlib import Path

from config import Settings
from runtime.failure_report import (
    clear_experiment_failure_report,
    experiment_failure_path,
    write_experiment_failure_report,
)


class TestExperimentFailureReport:
    """Failure report persistence."""

    def test_writes_report_when_experiment_active(self, tmp_path, monkeypatch) -> None:
        """Active experiment should write failure.json under .data/experiments/."""
        monkeypatch.chdir(tmp_path)
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test-model",
            langsearch_api_key="key",
            active_experiment="chess",
        )
        path = write_experiment_failure_report(
            settings,
            "context overflow",
            task_ids=["task-1"],
            cause="CriticalWakeError: context overflow",
        )
        assert path == Path(".data/experiments/chess/failure.json")
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == "failed"
        assert payload["experiment"] == "chess"
        assert payload["reason"] == "context overflow"
        assert payload["task_ids"] == ["task-1"]

    def test_clear_removes_existing_report(self, tmp_path, monkeypatch) -> None:
        """clear_experiment_failure_report should delete stale reports."""
        monkeypatch.chdir(tmp_path)
        settings = Settings(
            openai_base_url="http://localhost:8000/v1",
            openai_api_key="test",
            agent_model="test-model",
            langsearch_api_key="key",
            active_experiment="chess",
        )
        path = experiment_failure_path(settings)
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        clear_experiment_failure_report(settings)
        assert not path.exists()
