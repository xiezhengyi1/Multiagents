from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from shared.contracts import (
    AgentMessage,
    AgentTrace,
    ExecutionScope,
    ToolCall,
    ToolCatalog,
    ToolContract,
    TraceMetadata,
    TraceTarget,
    validate_message_sequence,
)


def _scope() -> ExecutionScope:
    return ExecutionScope(
        session_id="session-1",
        thread_id="session-1",
        snapshot_id="snapshot-1",
        round_index=1,
        scenario_id="scenario-1",
        scenario_tags=["training"],
        supi="imsi-208930000000001",
    )


def _tool_catalog() -> ToolCatalog:
    return ToolCatalog.create(
        [
            ToolContract(
                name="lookup_flow",
                description="Read the canonical flow binding for one SUPI and flow id.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "supi": {"type": "string"},
                        "flow_id": {"type": "string"},
                    },
                    "required": ["supi", "flow_id"],
                    "additionalProperties": False,
                },
                capabilities=["flow.read"],
                side_effect="read",
                result_format="json",
            )
        ]
    )


def _messages() -> list[AgentMessage]:
    return [
        AgentMessage(role="system", content="Ground the requested flow."),
        AgentMessage(role="user", content="Protect flow-1."),
        AgentMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="lookup_flow",
                    arguments={"supi": "imsi-208930000000001", "flow_id": "flow-1"},
                )
            ],
        ),
        AgentMessage(
            role="tool",
            name="lookup_flow",
            tool_call_id="call-1",
            tool_status="success",
            content='{"flow_id":"flow-1"}',
        ),
        AgentMessage(role="assistant", content='{"flow_id":"flow-1"}'),
    ]


def test_execution_scope_rejects_unknown_fields_and_duplicate_tags() -> None:
    with pytest.raises(ValidationError):
        ExecutionScope(
            session_id="session-1",
            thread_id="session-1",
            scenario_tags=["training", "training"],
        )
    with pytest.raises(ValidationError):
        ExecutionScope(session_id="session-1", thread_id="session-1", legacy_id="x")
    with pytest.raises(ValidationError):
        ExecutionScope(session_id="   ", thread_id="session-1")

    normalized = ExecutionScope(session_id=" session-1 ", thread_id=" thread-1 ")
    assert normalized.session_id == "session-1"
    assert normalized.thread_id == "thread-1"


def test_message_sequence_preserves_native_tool_call_order() -> None:
    validate_message_sequence(_messages(), require_final_assistant=True)

    invalid = [*_messages()[:3], _messages()[-1]]
    with pytest.raises(ValueError, match="pending tool calls"):
        validate_message_sequence(invalid, require_final_assistant=True)


def test_message_sequence_rejects_orphan_results_and_midstream_system_messages() -> None:
    orphan_result = [
        AgentMessage(role="system", content="System contract."),
        AgentMessage(role="user", content="Run lookup."),
        AgentMessage(
            role="tool",
            name="lookup_flow",
            tool_call_id="call-1",
            tool_status="success",
            content="{}",
        ),
    ]
    with pytest.raises(ValueError, match="no pending call"):
        validate_message_sequence(orphan_result, require_final_assistant=False)

    duplicate_system = [
        AgentMessage(role="system", content="System contract."),
        AgentMessage(role="user", content="Run lookup."),
        AgentMessage(role="system", content="Replacement system contract."),
    ]
    with pytest.raises(ValueError, match="only appear at the start"):
        validate_message_sequence(duplicate_system, require_final_assistant=False)


def test_tool_catalog_fingerprint_is_content_addressed() -> None:
    catalog = _tool_catalog()
    assert len(catalog.fingerprint) == 64
    with pytest.raises(ValidationError, match="fingerprint"):
        ToolCatalog(tools=catalog.tools, fingerprint="wrong")

    with pytest.raises(ValidationError, match="not valid under any of the given schemas"):
        ToolContract(
            name="invalid_schema",
            description="Expose an invalid schema for contract validation.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "not-a-json-schema-type"}},
            },
            side_effect="read",
            result_format="json",
        )


def test_successful_observed_trace_requires_exact_observed_target() -> None:
    now = datetime.now(timezone.utc)
    trace = AgentTrace(
        trace_id="trace-1",
        agent_name="intent_encoding",
        model_name="qwen3",
        source="observed_llm",
        scope=_scope(),
        started_at=now,
        ended_at=now,
        tool_catalog=_tool_catalog(),
        response_schema_name="GroundingAdvisorOutput",
        response_schema={"type": "object", "properties": {}},
        messages=_messages(),
        targets=[
            TraceTarget(
                kind="observed_model_output",
                producer="model",
                schema_name="GroundingAdvisorOutput",
                output_schema={"type": "object", "properties": {}},
                value={"flow_id": "flow-1"},
            ),
            TraceTarget(
                kind="validated_output",
                producer="validator",
                schema_name="GroundingDecision",
                output_schema={"type": "object", "properties": {}},
                value={"flow_id": "flow-1"},
            ),
        ],
        status="success",
        metadata=TraceMetadata(
            path_label="advisor_path",
            prompt_name="grounding.system",
            prompt_version="2.0.0",
        ),
    )
    assert trace.targets[0].sft_eligible is True
    assert trace.targets[1].sft_eligible is False


def test_target_sft_eligibility_is_derived_and_cannot_be_supplied() -> None:
    target = TraceTarget(
        kind="compiled_output",
        producer="compiler",
        schema_name="PolicyPlanDraft",
        output_schema={"type": "object"},
        value={"policies": []},
    )
    assert target.sft_eligible is False
    assert "sft_eligible" not in target.model_dump(mode="json")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TraceTarget(
            kind="compiled_output",
            producer="compiler",
            schema_name="PolicyPlanDraft",
            output_schema={"type": "object"},
            value={"policies": []},
            sft_eligible=True,
        )
