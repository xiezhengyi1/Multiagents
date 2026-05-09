from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["IntentEncodingAgent"]


def __getattr__(name: str) -> Any:
    if name == "IntentEncodingAgent":
        return import_module(".agent", __name__).IntentEncodingAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
