"""Tests for ThoughtBuddy configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from config import Settings


class TestSettings:
    """Settings loading tests."""

    def test_load_uses_yaml_and_model_defaults(self, tmp_path: Path, monkeypatch) -> None:
        """Load non-secret values from YAML when secrets are absent."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "logging": {"level": "DEBUG"},
                    "runtime": {"heartbeat_interval_seconds": 45},
                },
            ),
            encoding="utf-8",
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LANGSEARCH_API_KEY", raising=False)

        settings = Settings.load(config_path=config_path, env_file="/dev/null")

        assert settings.log_level == "DEBUG"
        assert settings.retry_log_each_attempt is False
        assert settings.retry_max_attempts == 4
        assert settings.llm_max_retries == 2
        assert settings.heartbeat_interval_seconds == 45
        assert settings.openai_base_url == "http://localhost:8000/v1"
        assert settings.openai_api_key == "EMPTY"

    def test_load_merges_secrets_from_env(self, tmp_path: Path, monkeypatch) -> None:
        """Read API keys exclusively from environment variables."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("logging:\n  level: INFO\n", encoding="utf-8")
        monkeypatch.setenv("OPENAI_API_KEY", "secret")
        monkeypatch.setenv("LANGSEARCH_API_KEY", "langsearch-secret")

        settings = Settings.load(config_path=config_path, env_file="/dev/null")

        assert settings.openai_api_key == "secret"
        assert settings.langsearch_api_key == "langsearch-secret"
        assert settings.log_level == "INFO"

    def test_load_missing_yaml_uses_model_defaults(self, monkeypatch) -> None:
        """Fall back to pydantic defaults when config.yaml is absent."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LANGSEARCH_API_KEY", raising=False)

        settings = Settings.load(
            config_path=Path("/nonexistent/config.yaml"),
            env_file="/dev/null",
        )

        assert settings.agent_model == "Qwen/Qwen3-8B-AWQ"
        assert settings.default_thread_id == "thoughtbuddy-main"
        assert settings.llm_temperature == 0.2
