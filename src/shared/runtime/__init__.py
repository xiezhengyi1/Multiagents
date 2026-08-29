from __future__ import annotations

"""Canonical shared implementation of agent-runtime primitives.

Application code should import runtime services from this package.  It is the
only supported runtime namespace.
"""

from .core import (
    AgentRuntimeContext,
    AgentWorkspace,
    ContextPolicy,
    RuntimeCache,
    TokenBudget,
    TokenCounter,
    project_root,
    runtime_root,
)
from .execution import StructuredToolLoop, ToolLoopExecutionError
from .messages import (
    build_tool_specs,
    extract_tool_calls,
    extract_tool_results,
    format_tool_call,
    format_tool_result,
    json_friendly,
    normalize_message_role,
    serialize_base_model_schema,
    serialize_message,
    stringify_message_content,
)
from .storage import (
    ArtifactCache,
    ArtifactEnvelope,
    ArtifactStore,
    FileTaskQueue,
    RuntimePersistence,
    SqlAlchemyRuntimePersistence,
)
from .trace import (
    JsonlTraceWriter,
    RunTreeEvent,
    RunTreeTraceRecord,
    TRACE_UNSET,
    TracedStructuredAgent,
    build_run_tree_record,
    collect_descendant_ids,
    dotted_order_key,
    iter_runs_in_dotted_order,
    utc_now,
)
from .worker import ArtifactWorkerMixin

__all__ = [
    "AgentRuntimeContext",
    "AgentWorkspace",
    "ArtifactCache",
    "ArtifactEnvelope",
    "ArtifactStore",
    "ArtifactWorkerMixin",
    "ContextPolicy",
    "FileTaskQueue",
    "JsonlTraceWriter",
    "RunTreeEvent",
    "RunTreeTraceRecord",
    "RuntimeCache",
    "StructuredToolLoop",
    "TRACE_UNSET",
    "TokenBudget",
    "TokenCounter",
    "ToolLoopExecutionError",
    "TracedStructuredAgent",
    "build_run_tree_record",
    "build_tool_specs",
    "collect_descendant_ids",
    "dotted_order_key",
    "extract_tool_calls",
    "extract_tool_results",
    "format_tool_call",
    "format_tool_result",
    "iter_runs_in_dotted_order",
    "json_friendly",
    "normalize_message_role",
    "project_root",
    "RuntimePersistence",
    "SqlAlchemyRuntimePersistence",
    "runtime_root",
    "serialize_base_model_schema",
    "serialize_message",
    "stringify_message_content",
    "utc_now",
]
