from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AgentRuntimeContext",
    "AgentWorkspace",
    "ArtifactCache",
    "ArtifactEnvelope",
    "ArtifactStore",
    "ContextPolicy",
    "FileTaskQueue",
    "JsonlTraceWriter",
    "RunTreeEvent",
    "RunTreeTraceRecord",
    "RuntimeCache",
    "StructuredToolLoop",
    "ToolLoopExecutionError",
    "TracedStructuredAgent",
    "build_tool_specs",
    "extract_tool_calls",
    "extract_tool_results",
    "format_tool_call",
    "format_tool_result",
    "json_friendly",
    "normalize_message_role",
    "project_trace_to_chatml_messages",
    "project_trace_to_training_trace",
    "runtime_root",
    "serialize_message",
    "stringify_message_content",
]


def __getattr__(name: str) -> Any:
    if name in {"AgentRuntimeContext", "AgentWorkspace", "ContextPolicy", "RuntimeCache", "runtime_root", "TokenBudget", "TokenCounter"}:
        return getattr(import_module(".core", __name__), name)
    if name in {"StructuredToolLoop", "ToolLoopExecutionError"}:
        return getattr(import_module(".execution.structured_tool_loop", __name__), name)
    if name in {
        "build_tool_specs",
        "extract_tool_calls",
        "extract_tool_results",
    }:
        return getattr(import_module(".tooling", __name__), name)
    if name in {
        "format_tool_call",
        "format_tool_result",
        "json_friendly",
        "normalize_message_role",
        "serialize_message",
        "stringify_message_content",
    }:
        return getattr(import_module(".messages", __name__), name)
    if name in {"ArtifactCache", "ArtifactEnvelope", "ArtifactStore", "FileTaskQueue"}:
        return getattr(import_module(".storage", __name__), name)
    if name in {"JsonlTraceWriter", "RunTreeEvent", "RunTreeTraceRecord", "TracedStructuredAgent"}:
        return getattr(import_module(".trace", __name__), name)
    if name in {"project_trace_to_chatml_messages", "project_trace_to_training_trace"}:
        return getattr(import_module(".trace.projectors", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
