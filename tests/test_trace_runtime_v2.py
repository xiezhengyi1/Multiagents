from __future__ import annotations

import json
from datetime import datetime, timezone

from pydantic import BaseModel

from shared.contracts import (
    AgentInvocation,
    AgentMessage,
    AgentRunResult,
    AgentTrace,
    ExecutionScope,
    ToolCall,
    ToolCatalog,
    ToolContract,
    TraceMetadata,
    TraceTarget,
)
from shared.runtime import AgentRuntimeContext, JsonlTraceWriter, ToolLoopExecutionError
from shared.runtime.trace import build_agent_trace


class _Decision(BaseModel):
    accepted: bool


def _scope() -> ExecutionScope:
    return ExecutionScope(
        session_id="session-1",
        thread_id="thread-1",
        snapshot_id="snapshot-1",
        round_index=1,
    )


def _invocation() -> AgentInvocation:
    return AgentInvocation(
        messages=[AgentMessage(role="user", content="Evaluate the request.")],
        metadata=TraceMetadata(
            path_label="decision",
            prompt_name="decision_prompt",
            prompt_version="2.0.0",
        ),
    )


def test_success_trace_round_trips_through_jsonl_writer(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    catalog = ToolCatalog.create([])
    result = AgentRunResult(
        messages=[AgentMessage(role="assistant", content='{"accepted":true}')],
        structured_response={"accepted": True},
    )
    trace = build_agent_trace(
        agent_name="decision_agent",
        model_name="qwen3",
        system_prompt="Return one decision JSON object.",
        tool_catalog=catalog,
        response_model=_Decision,
        invocation=_invocation(),
        context=AgentRuntimeContext(agent_name="decision_agent", scope=_scope()),
        result=result,
        captured_error=None,
        started_at=now,
        ended_at=now,
        derived_targets=[
            TraceTarget(
                kind="validated_output",
                producer="validator",
                schema_name=_Decision.__name__,
                output_schema=_Decision.model_json_schema(),
                value={"accepted": True},
                sft_eligible=False,
            )
        ],
    )

    path = JsonlTraceWriter("decision_agent", root=tmp_path).write(trace)
    rows = path.read_text(encoding="utf-8").splitlines()

    assert len(rows) == 1
    restored = AgentTrace.model_validate(json.loads(rows[0]))
    assert restored == trace
    assert [target.kind for target in restored.targets] == [
        "observed_model_output",
        "validated_output",
    ]


def test_tool_failure_trace_keeps_unresolved_call_as_error_evidence() -> None:
    now = datetime.now(timezone.utc)
    catalog = ToolCatalog.create(
        [
            ToolContract(
                name="lookup_state",
                description="Read one state object.",
                input_schema={
                    "type": "object",
                    "properties": {"state_id": {"type": "string"}},
                    "required": ["state_id"],
                    "additionalProperties": False,
                },
                side_effect="read",
                result_format="json",
            )
        ]
    )
    failed_call = ToolCall(
        id="call-1",
        name="lookup_state",
        arguments={"state_id": "state-1"},
    )
    captured_error = ToolLoopExecutionError(
        "lookup_state failed",
        output_messages=[AgentMessage(role="assistant", tool_calls=[failed_call])],
        failed_tool_call={
            "id": failed_call.id,
            "name": failed_call.name,
            "args": failed_call.arguments,
        },
    )

    trace = build_agent_trace(
        agent_name="decision_agent",
        model_name="qwen3",
        system_prompt="Use declared tools and return one decision JSON object.",
        tool_catalog=catalog,
        response_model=_Decision,
        invocation=_invocation(),
        context=AgentRuntimeContext(agent_name="decision_agent", scope=_scope()),
        result=None,
        captured_error=captured_error,
        started_at=now,
        ended_at=now,
    )

    assert trace.status == "error"
    assert trace.error == "lookup_state failed"
    assert trace.targets == []
    assert trace.messages[-1].tool_calls == [failed_call]
