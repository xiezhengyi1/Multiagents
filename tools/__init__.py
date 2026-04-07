"""Backward-compatible aliases for legacy `tools` imports.

This module maps `tools.*` to `agents.tools.*` so older imports and test
patch targets continue to work after package refactoring.
"""

from __future__ import annotations

import importlib
import sys

from agents.tools import *  # noqa: F403
from agents.tools import __all__ as _agents_tools_all

_MODULE_ALIASES = (
    "knowledge_tool",
    "pcf_tools",
    "db_tool",
    "network_status",
    "network_graph",
    "dashboard_data",
    "init_scenario",
    "session_context_tool",
    "wrapper_think",
    "optimizer",
)

for _name in _MODULE_ALIASES:
    _module = importlib.import_module(f"agents.tools.{_name}")
    sys.modules[f"tools.{_name}"] = _module

__all__ = list(_agents_tools_all)


def __getattr__(name: str):
    if name in _MODULE_ALIASES:
        module = importlib.import_module(f"agents.tools.{name}")
        sys.modules[f"tools.{name}"] = module
        return module
    raise AttributeError(f"module 'tools' has no attribute {name!r}")
