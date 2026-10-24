"""Tests for generic experiment runner and chess setup entrypoint."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.chess.setup import apply_settings, preflight
from experiments.runner.apply import apply_to_app_config
from experiments.runner.config import ExperimentRunConfig, shell_env
from experiments.runner.entrypoint import SetupError, run_preflight
from experiments.runner.llm_profiles import resolve_llm_profile


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """Minimal repo layout with app config and experiment config."""
    app_config = tmp_path / "config.yaml"
    app_config.write_text(
        """llm:
  profile: qwen3
  agent_model: Qwen/Qwen3-8B-AWQ
  enable_thinking: false
  context_window_tokens: 32768
experiment:
  active: chess
  chess:
    engine_path: .data/stockfish/engine
    agent_color: white
    agent_generation: 0
    skill_level: 3
    move_time_seconds: 0.1
    engine_timeout_seconds: 10
""",
        encoding="utf-8",
    )

    exp_config = tmp_path / "config/experiments/chess-lfm.yaml"
    exp_config.parent.mkdir(parents=True)
    exp_config.write_text(
        """experiment: chess
entrypoint: experiments.chess.setup
llm:
  profile: lfm
stack:
  thoughtbuddy_port: 3030
  vllm_port: 9000
paths:
  run_dir: .data/experiments/chess-lfm
app_config: config.yaml
chess:
  agent_generation: 2
""",
        encoding="utf-8",
    )

    engine = tmp_path / ".data/stockfish/engine"
    engine.parent.mkdir(parents=True)
    engine.write_text("#!/bin/sh\n", encoding="utf-8")
    engine.chmod(0o755)
    return tmp_path


class TestExperimentRunConfig:
    """Experiment YAML loading."""

    def test_loads_entrypoint_and_settings(self, repo_root: Path) -> None:
        """Load entrypoint and experiment-specific settings block."""
        config = ExperimentRunConfig.load(
            repo_root / "config/experiments/chess-lfm.yaml",
            repo_root=repo_root,
        )
        assert config.experiment == "chess"
        assert config.entrypoint == "experiments.chess.setup"
        assert config.llm_profile == "lfm"
        assert config.experiment_settings["agent_generation"] == 2


class TestApplyToAppConfig:
    """Sync experiment config into config.yaml."""

    def test_patches_llm_and_delegates_chess_settings(self, repo_root: Path) -> None:
        """Apply lfm profile and chess overrides via setup entrypoint."""
        run_config = ExperimentRunConfig.load(
            repo_root / "config/experiments/chess-lfm.yaml",
            repo_root=repo_root,
        )
        app_path = apply_to_app_config(run_config, repo_root=repo_root)
        data = yaml.safe_load(app_path.read_text(encoding="utf-8"))
        assert data["llm"]["profile"] == "lfm"
        assert data["llm"]["agent_model"] == "LiquidAI/LFM2.5-8B-A1B"
        assert data["experiment"]["chess"]["agent_generation"] == 2


class TestChessSetup:
    """Chess setup entrypoint."""

    def test_apply_settings_patches_chess_block(self, repo_root: Path) -> None:
        """Patch only experiment.chess keys."""
        run_config = ExperimentRunConfig.load(
            repo_root / "config/experiments/chess-lfm.yaml",
            repo_root=repo_root,
        )
        text = (repo_root / "config.yaml").read_text(encoding="utf-8")
        updated = apply_settings(run_config, text)
        data = yaml.safe_load(updated)
        assert data["experiment"]["chess"]["agent_generation"] == 2

    def test_preflight_requires_stockfish(self, repo_root: Path) -> None:
        """Pass when engine binary exists."""
        run_config = ExperimentRunConfig.load(
            repo_root / "config/experiments/chess-lfm.yaml",
            repo_root=repo_root,
        )
        preflight(run_config, repo_root=repo_root)

    def test_preflight_fails_when_engine_missing(self, repo_root: Path) -> None:
        """Fail when Stockfish is missing."""
        run_config = ExperimentRunConfig.load(
            repo_root / "config/experiments/chess-lfm.yaml",
            repo_root=repo_root,
        )
        (repo_root / ".data/stockfish/engine").unlink()
        with pytest.raises(SetupError, match="Stockfish"):
            run_preflight(run_config, repo_root=repo_root)


class TestShellEnv:
    """Shell environment generation."""

    def test_exports_vllm_profile_and_experiment(self, repo_root: Path) -> None:
        """Emit VLLM_PROFILE and experiment metadata for run.sh."""
        run_config = ExperimentRunConfig.load(
            repo_root / "config/experiments/chess-lfm.yaml",
            repo_root=repo_root,
        )
        env = shell_env(run_config, repo_root=repo_root)
        assert env["VLLM_PROFILE"] == "lfm"
        assert env["EXPERIMENT_NAME"] == "chess"
        assert env["EXPERIMENT_ENTRYPOINT"] == "experiments.chess.setup"
        assert env["TB_PORT"] == "3030"


class TestLlmProfiles:
    """LLM profile registry."""

    def test_lfm_profile_context_window(self) -> None:
        """LFM profile uses 128K context window."""
        profile = resolve_llm_profile("lfm")
        assert profile["context_window_tokens"] == 131072
