"""Chess experiment plugin — virtual board tools, HTTP UI, and task materialization."""

from __future__ import annotations

from experiments.chess.plugin.tools import create_tools
from experiments.registry import register_experiment

register_experiment("chess", create_tools)

__all__ = ["create_tools"]
