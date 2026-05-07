from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "NetworkGraph": ".network_graph",
    "cache_scenario": ".common",
    "get_cached_control_scenario": ".common",
    "get_current_scenario": ".init_scenario",
    "get_initial_scenario": ".init_scenario",
    "get_current_optimizer_scenario": ".init_scenario",
    "get_graph_snapshot_payload": ".network_graph",
    "get_latest_graph": ".network_graph",
    "get_latest_graph_snapshot_metadata": ".network_graph",
    "init_main": ".init_scenario",
    "initialize_scenario": ".init_scenario",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name, __name__), name)
