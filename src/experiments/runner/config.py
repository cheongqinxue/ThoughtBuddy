"""Load generic experiment run configuration from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field

from experiments.runner.llm_profiles import resolve_llm_profile


class StackConfig(BaseModel):
    """Process stack settings for run.sh."""

    thoughtbuddy_port: int = Field(default=2024, ge=1, le=65535)
    vllm_port: int = Field(default=8000, ge=1, le=65535)
    vllm_ready_timeout_seconds: int = Field(default=600, ge=1)
    thoughtbuddy_ready_timeout_seconds: int = Field(default=120, ge=1)
    monitor_interval_seconds: int = Field(default=5, ge=1)
    skip_langfuse: bool = False
    stop_langfuse_on_teardown: bool = True


class PathsConfig(BaseModel):
    """Runtime directories for logs and PID files."""

    run_dir: Path = Field(default=Path(".data/experiments/default"))


class ExperimentRunConfig(BaseModel):
    """Experiment run configuration from YAML."""

    experiment: str
    entrypoint: str
    llm: dict[str, str] = Field(default_factory=lambda: {"profile": "qwen3"})
    stack: StackConfig = Field(default_factory=StackConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    app_config: Path = Field(default=Path("config.yaml"))
    config_path: Path | None = Field(default=None, exclude=True)
    experiment_settings: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def llm_profile(self) -> str:
        """Named LLM profile (qwen3 or lfm)."""
        profile = self.llm.get("profile", "qwen3")
        if not isinstance(profile, str):
            msg = "llm.profile must be a string"
            raise TypeError(msg)
        return profile

    @property
    def run_dir(self) -> Path:
        """Repo-relative run directory."""
        return self.paths.run_dir

    def resolved_paths(self, repo_root: Path) -> dict[str, Path]:
        """Resolve run paths relative to the repository root.

        Args:
            repo_root: ThoughtBuddy repository root.

        Returns:
            Named paths for shell orchestration.
        """
        run_dir = (repo_root / self.run_dir).resolve()
        log_dir = run_dir / "logs"
        return {
            "run_dir": run_dir,
            "log_dir": log_dir,
            "pid_dir": run_dir / "pids",
            "manifest": run_dir / "manifest.env",
            "failure_report": run_dir / "failure.json",
            "vllm_log": log_dir / "vllm.log",
            "thoughtbuddy_log": log_dir / "thoughtbuddy.log",
            "langfuse_log": log_dir / "langfuse.log",
            "monitor_log": log_dir / "monitor.log",
        }

    @classmethod
    def load(cls, config_path: Path, *, repo_root: Path | None = None) -> Self:
        """Load experiment run config from YAML.

        Args:
            config_path: Path to experiment YAML.
            repo_root: Repository root for resolving relative paths.

        Returns:
            Validated experiment run configuration.

        Raises:
            FileNotFoundError: When the config file is missing.
            ValueError: When YAML content is invalid.
        """
        root = repo_root or Path.cwd()
        path = config_path if config_path.is_absolute() else root / config_path
        if not path.is_file():
            msg = f"experiment config not found: {path}"
            raise FileNotFoundError(msg)

        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        if not isinstance(raw, dict):
            msg = f"experiment config must be a mapping: {path}"
            raise ValueError(msg)

        experiment = raw.get("experiment")
        if not isinstance(experiment, str) or not experiment.strip():
            msg = "experiment config requires string field 'experiment'"
            raise ValueError(msg)

        entrypoint = raw.get("entrypoint")
        if not isinstance(entrypoint, str) or not entrypoint.strip():
            msg = "experiment config requires string field 'entrypoint'"
            raise ValueError(msg)

        known_keys = {
            "experiment",
            "entrypoint",
            "llm",
            "stack",
            "paths",
            "app_config",
        }
        settings_block = raw.get(experiment)
        experiment_settings = settings_block if isinstance(settings_block, dict) else {}

        payload = {
            key: value for key, value in raw.items() if key in known_keys
        }
        payload["experiment"] = experiment
        payload["entrypoint"] = entrypoint
        payload["config_path"] = path.resolve()
        payload["experiment_settings"] = experiment_settings

        return cls.model_validate(payload)


def shell_env(run_config: ExperimentRunConfig, *, repo_root: Path) -> dict[str, str]:
    """Build environment variables for run.sh.

    Args:
        run_config: Loaded experiment run configuration.
        repo_root: Repository root.

    Returns:
        String key/value pairs safe for shell export.
    """
    paths = run_config.resolved_paths(repo_root)
    llm = resolve_llm_profile(run_config.llm_profile)
    return {
        "EXP_CONFIG": str(run_config.config_path or ""),
        "EXPERIMENT_NAME": run_config.experiment,
        "EXPERIMENT_ENTRYPOINT": run_config.entrypoint,
        "VLLM_PROFILE": llm["profile"],
        "TB_PORT": str(run_config.stack.thoughtbuddy_port),
        "VLLM_PORT": str(run_config.stack.vllm_port),
        "VLLM_READY_TIMEOUT": str(run_config.stack.vllm_ready_timeout_seconds),
        "TB_READY_TIMEOUT": str(run_config.stack.thoughtbuddy_ready_timeout_seconds),
        "MONITOR_INTERVAL": str(run_config.stack.monitor_interval_seconds),
        "RUN_DIR": str(paths["run_dir"]),
        "LOG_DIR": str(paths["log_dir"]),
        "PID_DIR": str(paths["pid_dir"]),
        "MANIFEST": str(paths["manifest"]),
        "FAILURE_REPORT": str(paths["failure_report"]),
        "VLLM_LOG": str(paths["vllm_log"]),
        "TB_LOG": str(paths["thoughtbuddy_log"]),
        "LANGFUSE_LOG": str(paths["langfuse_log"]),
        "MONITOR_LOG": str(paths["monitor_log"]),
        "SKIP_LANGFUSE": "1" if run_config.stack.skip_langfuse else "0",
        "STOP_LANGFUSE": "1" if run_config.stack.stop_langfuse_on_teardown else "0",
        "LLM_AGENT_MODEL": llm["agent_model"],
    }
