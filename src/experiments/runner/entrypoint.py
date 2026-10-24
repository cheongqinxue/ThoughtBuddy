"""Dispatch to experiment setup entrypoint modules."""

from __future__ import annotations

import importlib
from typing import Any

from experiments.runner.config import ExperimentRunConfig


class SetupError(RuntimeError):
    """Raised when experiment setup checks fail."""


def load_setup_module(entrypoint: str):
    """Import an experiment setup module.

    Args:
        entrypoint: Dotted module path.

    Returns:
        Imported module.
    """
    return importlib.import_module(entrypoint)


def run_preflight(run_config: ExperimentRunConfig, *, repo_root: Any) -> None:
    """Run experiment-specific preflight checks.

    Args:
        run_config: Loaded experiment run configuration.
        repo_root: Repository root path.

    Raises:
        SetupError: When preflight fails.
    """
    setup = load_setup_module(run_config.entrypoint)
    preflight = getattr(setup, "preflight", None)
    if preflight is None:
        msg = f"{run_config.entrypoint} must define preflight()"
        raise AttributeError(msg)
    try:
        preflight(run_config, repo_root=repo_root)
    except SetupError:
        raise
    except Exception as exc:
        raise SetupError(str(exc)) from exc


def run_analyze(run_config: ExperimentRunConfig, *, repo_root: Any) -> None:
    """Print experiment-specific analysis output.

    Args:
        run_config: Loaded experiment run configuration.
        repo_root: Repository root path.
    """
    setup = load_setup_module(run_config.entrypoint)
    analyze = getattr(setup, "analyze", None)
    if analyze is None:
        msg = f"{run_config.entrypoint} must define analyze()"
        raise AttributeError(msg)
    analyze(run_config, repo_root=repo_root)


def start_hints(run_config: ExperimentRunConfig, *, tb_port: int) -> list[str]:
    """Return optional startup banner lines for run.sh.

    Args:
        run_config: Loaded experiment run configuration.
        tb_port: ThoughtBuddy HTTP port.

    Returns:
        Lines to print after the stack starts (may be empty).
    """
    setup = load_setup_module(run_config.entrypoint)
    hints_fn = getattr(setup, "start_hints", None)
    if hints_fn is None:
        return []
    return list(hints_fn(run_config, tb_port=tb_port))
