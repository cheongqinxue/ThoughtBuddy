"""Registry for experiment tool factories."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import BaseTool

from config import Settings

ExperimentFactory = Callable[[Settings], list[BaseTool]]

_FACTORIES: dict[str, ExperimentFactory] = {}


def register_experiment(name: str, factory: ExperimentFactory) -> None:
    """Register an experiment that exposes agent tools.

    Args:
        name: Experiment identifier (e.g. ``chess``).
        factory: Callable that takes ``Settings`` and returns LangChain tools.
    """
    _FACTORIES[name] = factory


def list_experiments() -> list[str]:
    """Return registered experiment names in registration order.

    Returns:
        Sorted experiment identifiers.
    """
    return sorted(_FACTORIES)


def get_experiment_tools(settings: Settings) -> list[BaseTool]:
    """Return tools for the active experiment, if any.

    Args:
        settings: Application settings.

    Returns:
        Experiment tools, or an empty list when no experiment is active.

    Raises:
        ValueError: When ``active_experiment`` is set but not registered.
    """
    name = settings.active_experiment.strip()
    if not name:
        return []

    factory = _FACTORIES.get(name)
    if factory is None:
        msg = f"Unknown experiment: {name!r}. Registered: {', '.join(list_experiments()) or '(none)'}"
        raise ValueError(msg)
    return factory(settings)
