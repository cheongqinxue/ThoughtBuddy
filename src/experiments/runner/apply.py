"""Patch config.yaml from experiment run config."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

from experiments.runner.config import ExperimentRunConfig
from experiments.runner.llm_profiles import resolve_llm_profile


def _patch_scalar(text: str, *, key: str, value: Any, indent: int) -> str:
    """Replace one ``key: value`` line under a fixed indent.

    Args:
        text: Full config file text.
        key: YAML key to replace.
        value: New scalar value.
        indent: Leading spaces before the key.

    Returns:
        Updated file text.

    Raises:
        ValueError: When the key line is not found.
    """
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = str(value)
    spaces = " " * indent
    pattern = rf"(^{spaces}{re.escape(key)}:\s*)(.+)$"
    new_text, count = re.subn(
        pattern,
        rf"\g<1>{rendered}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 0:
        msg = f"config.yaml missing key {key!r} (expected {indent} spaces indent)"
        raise ValueError(msg)
    return new_text


def _load_entrypoint(module_path: str):
    """Import an experiment setup module.

    Args:
        module_path: Dotted module path (e.g. ``experiments.chess.setup``).

    Returns:
        Imported module.
    """
    return importlib.import_module(module_path)


def apply_to_app_config(run_config: ExperimentRunConfig, *, repo_root: Path) -> Path:
    """Sync experiment YAML into the main ThoughtBuddy config.yaml.

    Patches ``llm.*`` and ``experiment.active``, then delegates experiment-specific
    settings to the configured entrypoint module.

    Args:
        run_config: Loaded experiment run configuration.
        repo_root: Repository root.

    Returns:
        Path to the updated app config file.
    """
    app_path = (
        run_config.app_config
        if run_config.app_config.is_absolute()
        else repo_root / run_config.app_config
    )
    if not app_path.is_file():
        msg = f"app config not found: {app_path}"
        raise FileNotFoundError(msg)

    text = app_path.read_text(encoding="utf-8")
    llm_values = resolve_llm_profile(run_config.llm_profile)

    for key, value in llm_values.items():
        text = _patch_scalar(text, key=key, value=value, indent=2)

    text = _patch_scalar(text, key="active", value=run_config.experiment, indent=2)

    setup = _load_entrypoint(run_config.entrypoint)
    apply_settings = getattr(setup, "apply_settings", None)
    if apply_settings is None:
        msg = f"{run_config.entrypoint} must define apply_settings()"
        raise AttributeError(msg)
    text = apply_settings(run_config, text)

    app_path.write_text(text, encoding="utf-8")
    return app_path
