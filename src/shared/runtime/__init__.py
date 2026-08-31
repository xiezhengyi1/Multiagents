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
    build_tool_catalog,
    contract_to_message,
    extract_tool_calls,
    extract_tool_results,
    json_friendly,
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
    TracedStructuredAgent,
    build_agent_trace,
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
    "RuntimeCache",
    "StructuredToolLoop",
    "TokenBudget",
    "TokenCounter",
    "ToolLoopExecutionError",
    "TracedStructuredAgent",
    "build_agent_trace",
    "build_tool_catalog",
    "contract_to_message",
    "extract_tool_calls",
    "extract_tool_results",
    "json_friendly",
    "project_root",
    "RuntimePersistence",
    "SqlAlchemyRuntimePersistence",
    "runtime_root",
    "serialize_message",
    "stringify_message_content",
]
