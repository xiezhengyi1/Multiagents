from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AmPolicySpec",
    "OptimizationStrategyAgent",
    "OptimizationStrategyCompiler",
    "OsaAdvisorOutput",
    "SmPolicySpec",
    "UrspPolicySpec",
]


def __getattr__(name: str) -> Any:
    if name == "OptimizationStrategyAgent":
        return import_module(".agent", __name__).OptimizationStrategyAgent
    if name == "OptimizationStrategyCompiler":
        return import_module(".compiler", __name__).OptimizationStrategyCompiler
    if name in {"AmPolicySpec", "OsaAdvisorOutput", "SmPolicySpec", "UrspPolicySpec"}:
        return getattr(import_module(".response_models", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
