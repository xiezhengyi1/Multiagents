from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel

from shared.contracts import (
    AgentInvocation,
    AgentMessage,
    AgentRunResult,
    AgentTrace,
    ToolCatalog,
    TraceTarget,
)
from shared.runtime.core.context import AgentRuntimeContext
from shared.runtime.execution.structured_tool_loop import ToolLoopExecutionError
from shared.runtime.messages import json_friendly


def build_agent_trace(
    *,
    agent_name: str,
    model_name: str,
    system_prompt: str,
    tool_catalog: ToolCatalog,
    response_model: type[BaseModel],
    invocation: AgentInvocation,
    context: AgentRuntimeContext,
    result: AgentRunResult | None,
    captured_error: BaseException | None,
    started_at: datetime,
    ended_at: datetime,
    derived_targets: list[TraceTarget] | None = None,
) -> AgentTrace:
    """Build one canonical, model-independent trace record."""

    status = "error" if captured_error is not None else "success"
    if status == "success" and result is None:
        raise ValueError("successful trace requires AgentRunResult")

    output_messages = (
        list(captured_error.output_messages)
        if isinstance(captured_error, ToolLoopExecutionError)
        else list(result.messages if result is not None else [])
    )
    targets = list(derived_targets or [])
    if result is not None:
        targets.insert(
            0,
            TraceTarget(
                kind="observed_model_output",
                producer="model",
                schema_name=response_model.__name__,
                output_schema=json_friendly(response_model.model_json_schema()),
                value=result.structured_response,
            ),
        )

    return AgentTrace(
        trace_id=f"trace-{uuid4()}",
        agent_name=agent_name,
        model_name=model_name,
        source="observed_llm",
        scope=context.scope,
        started_at=started_at,
        ended_at=ended_at,
        tool_catalog=tool_catalog,
        response_schema_name=response_model.__name__,
        response_schema=json_friendly(response_model.model_json_schema()),
        messages=[
            AgentMessage(role="system", content=system_prompt),
            *invocation.messages,
            *output_messages,
        ],
        targets=targets,
        status=status,
        error=str(captured_error) if captured_error is not None else None,
        metadata=invocation.metadata,
    )


__all__ = ["build_agent_trace"]
