from __future__ import annotations

from importlib import import_module
from typing import Any

from agent_runtime.tooling import build_tool_specs, extract_tool_calls, extract_tool_results

__all__ = [
    "AgentRuntimeContext",
    "AgentWorkspace",
    "ArtifactCache",
    "ArtifactEnvelope",
    "ArtifactStore",
    "ArtifactWorkerMixin",
    "FileTaskQueue",
    "JsonlTraceWriter",
    "RuntimeCache",
    "StructuredToolLoop",
    "ToolLoopExecutionError",
    "TracedStructuredAgent",
    "build_run_tree_record",
    "build_tool_specs",
    "extract_tool_calls",
    "extract_tool_results",
]


def __getattr__(name: str) -> Any:
    if name == "ArtifactWorkerMixin":
        return import_module(".worker", __name__).ArtifactWorkerMixin
    if name == "build_run_tree_record":
        return import_module("agent_runtime.trace.builder").build_run_tree_record
    if name in {
        "AgentRuntimeContext",
        "AgentWorkspace",
        "ArtifactCache",
        "ArtifactEnvelope",
        "ArtifactStore",
        "FileTaskQueue",
        "JsonlTraceWriter",
        "RuntimeCache",
        "TracedStructuredAgent",
    }:
        return getattr(import_module("agent_runtime"), name)
    if name in {"StructuredToolLoop", "ToolLoopExecutionError"}:
        return getattr(import_module("agent_runtime.execution.structured_tool_loop"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
