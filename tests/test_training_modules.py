from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from shared.contracts import (
    AgentMessage,
    AgentTrace,
    ExecutionScope,
    ToolCall,
    ToolCatalog,
    ToolContract,
    TraceMetadata,
    TraceTarget,
)
from training.artifacts import read_jsonl, write_jsonl
from training.project import export_qwen3, project_agents
from training.projection import build_sft_example, build_workflow_record, project_trace
from training.schemas import Qwen3SftRecord, SftExampleV2


def _trace(*, trace_id: str = "trace-1", day: int = 1) -> AgentTrace:
    timestamp = datetime(2026, 1, day, tzinfo=timezone.utc)
    response = {"flow_id": "flow-1"}
    schema = {
        "type": "object",
        "properties": {"flow_id": {"type": "string"}},
        "required": ["flow_id"],
    }
    catalog = ToolCatalog.create(
        [
            ToolContract(
                name="lookup",
                description="Look up one flow.",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                side_effect="read",
                result_format="json",
            )
        ]
    )
    return AgentTrace(
        trace_id=trace_id,
        agent_name="intent_encoding",
        model_name="model-1",
        source="observed_llm",
        scope=ExecutionScope(
            session_id="session-1",
            thread_id="thread-1",
            snapshot_id="snapshot-1",
        ),
        started_at=timestamp,
        ended_at=timestamp,
        tool_catalog=catalog,
        response_schema_name="Decision",
        response_schema=schema,
        messages=[
            AgentMessage(role="system", content="Ground the request."),
            AgentMessage(role="user", content="protect flow-1"),
            AgentMessage(
                role="assistant",
                tool_calls=[ToolCall(id="call-1", name="lookup", arguments={"flow": "flow-1"})],
            ),
            AgentMessage(
                role="tool",
                content='{"flow_id":"flow-1"}',
                name="lookup",
                tool_call_id="call-1",
                tool_status="success",
            ),
            AgentMessage(role="assistant", content='{"flow_id":"flow-1"}'),
        ],
        targets=[
            TraceTarget(
                kind="observed_model_output",
                producer="model",
                schema_name="Decision",
                output_schema=schema,
                value=response,
            ),
            TraceTarget(
                kind="validated_output",
                producer="validator",
                schema_name="GroundingDecision",
                output_schema=schema,
                value=response,
            ),
        ],
        status="success",
        metadata=TraceMetadata(
            path_label="advisor_path",
            prompt_name="intent_grounding_advisor",
            prompt_version="2.0.0",
        ),
    )


def test_artifact_io_supports_overwrite_and_append(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"id": 1}])
    write_jsonl(path, [{"id": 2}], mode="a")
    assert read_jsonl(path) == [{"id": 1}, {"id": 2}]


def test_agent_projection_emits_validated_sft_v2(monkeypatch, tmp_path: Path) -> None:
    import training.project as project_module

    monkeypatch.setattr(
        project_module,
        "load_traces",
        lambda _agents: {"intent_encoding": [_trace()]},
    )
    counts = project_agents(agents=["intent_encoding"], output_dir=tmp_path)
    row = read_jsonl(tmp_path / "intent_encoding_sft_v2.jsonl")[0]
    example = SftExampleV2.model_validate(row)
    assert counts == {"intent_encoding": 1}
    assert example.observed_target.value == {"flow_id": "flow-1"}
    assert example.tool_catalog.tools[0].name == "lookup"


def test_qwen3_export_preserves_native_tool_order(tmp_path: Path) -> None:
    input_path = tmp_path / "sft_v2.jsonl"
    output_path = tmp_path / "qwen3.jsonl"
    write_jsonl(input_path, [build_sft_example(_trace())])

    assert export_qwen3(input_path=input_path, output_path=output_path) == 1
    record = Qwen3SftRecord.model_validate(read_jsonl(output_path)[0])
    assert [message["role"] for message in record.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert record.messages[2]["tool_calls"][0]["function"]["arguments"] == {"flow": "flow-1"}
    assert record.tools[0]["function"]["name"] == "lookup"


def test_workflow_projection_orders_and_flattens_agent_traces() -> None:
    later = project_trace(_trace(trace_id="trace-2", day=2))
    earlier = project_trace(_trace(trace_id="trace-1", day=1))
    record = build_workflow_record(
        {
            "session_id": "session-1",
            "snapshot_id": "snapshot-1",
            "status": "success",
            "completed": True,
            "user_input": "protect flow-1",
        },
        session_traces={"intent_encoding": [later, earlier]},
    )
    assert [trace.trace_id for trace in record.agent_trajectories["intent_encoding"]] == [
        "trace-1",
        "trace-2",
    ]
    assert [call["agent_name"] for call in record.tool_calls] == [
        "intent_encoding",
        "intent_encoding",
    ]
    assert record.scope.session_id == "session-1"
    assert record.scope.snapshot_id == "snapshot-1"
