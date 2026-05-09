from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["MainControlAgent"]


def __getattr__(name: str) -> Any:
    if name == "MainControlAgent":
        return import_module(".agent", __name__).MainControlAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
