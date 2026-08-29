from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from training.artifacts import read_jsonl, write_jsonl
from training.project import export_chatml, project_agents
from training.projection import build_workflow_record
from training.schemas import ChatmlSftRecord, FineTuningTraceRecord, ProjectedTraceRecord


def _trace(
    *,
    trace_id: str = "trace-1",
    timestamp: str = "2026-01-01T00:00:00Z",
) -> ProjectedTraceRecord:
    return ProjectedTraceRecord(
        trace_id=trace_id,
        timestamp=timestamp,
        agent_name="intent_encoding",
        session_id="session-1",
        snapshot_id="snapshot-1",
        thread_id="thread-1",
        model_name="model-1",
        input_messages=[{"role": "user", "content": "protect flow-1"}],
        tool_calls=[{"id": "call-1", "name": "lookup", "args": {"flow": "flow-1"}}],
        tool_results=[{"tool_call_id": "call-1", "name": "lookup", "content": "ok"}],
        structured_response={"flow_id": "flow-1"},
        status="success",
    )


def test_artifact_io_supports_overwrite_and_append(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"

    write_jsonl(path, [{"id": 1}])
    write_jsonl(path, [{"id": 2}], mode="a")

    assert read_jsonl(path) == [{"id": 1}, {"id": 2}]


def test_agent_projection_uses_one_finetuning_schema(monkeypatch, tmp_path: Path) -> None:
    import training.project as project_module

    monkeypatch.setattr(
        project_module,
        "load_traces",
        lambda _agents: {"intent_encoding": [_trace()]},
    )

    counts = project_agents(agents=["intent_encoding"], output_dir=tmp_path)
    row = read_jsonl(tmp_path / "intent_encoding_trajectories_v1.jsonl")[0]

    assert counts == {"intent_encoding": 1}
    assert FineTuningTraceRecord.model_validate(row).output == {"flow_id": "flow-1"}
    assert "message_trajectory" not in row


def test_chatml_export_consumes_finetuning_records(tmp_path: Path) -> None:
    input_path = tmp_path / "traces.jsonl"
    output_path = tmp_path / "chatml.jsonl"
    trace = FineTuningTraceRecord(
        trace_id="trace-1",
        timestamp="2026-01-01T00:00:00Z",
        agent_name="intent_encoding",
        session_id="session-1",
        snapshot_id="snapshot-1",
        input=[{"role": "user", "content": "protect flow-1"}],
        output={"flow_id": "flow-1"},
        status="success",
    )
    write_jsonl(input_path, [trace])

    assert export_chatml(input_path=input_path, output_path=output_path) == 1
    record = ChatmlSftRecord.model_validate(read_jsonl(output_path)[0])
    assert record.messages[0].role == "user"
    assert record.messages[-1].role == "assistant"


def test_workflow_projection_orders_and_flattens_agent_traces() -> None:
    later = _trace(trace_id="trace-2", timestamp="2026-01-02T00:00:00Z")
    earlier = _trace(trace_id="trace-1", timestamp="2026-01-01T00:00:00Z")

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

    assert [trace.trace_id for trace in record.agent_trajectories["intent_encoding"]] == ["trace-1", "trace-2"]
    assert [call["agent_name"] for call in record.tool_calls] == ["intent_encoding", "intent_encoding"]
