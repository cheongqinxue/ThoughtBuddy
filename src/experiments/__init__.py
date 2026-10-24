"""Experiment plugins that register virtual-world tools for the agent."""

from __future__ import annotations

import experiments.chess.plugin as _chess_plugin  # noqa: F401 — registers chess on import
from experiments.registry import get_experiment_tools, list_experiments, register_experiment

__all__ = ["get_experiment_tools", "list_experiments", "register_experiment"]
