from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from control_runtime.agents.dispatch.agent import PolicyDispatchAgent
from shared.contracts import AgentTrace, ArtifactEnvelope, ExecutionScope
from shared.runtime.trace.execution import write_deterministic_execution_trace


class _RecordingWriter:
    def __init__(self) -> None:
        self.records: list[AgentTrace] = []

    def write(self, record: AgentTrace) -> None:
        self.records.append(record)


class _RecordingCache:
    def __init__(self) -> None:
        self.produced: list[ArtifactEnvelope] = []

    def cache_produced(self, envelope: ArtifactEnvelope) -> None:
        self.produced.append(envelope)


def _request_envelope() -> ArtifactEnvelope:
    return ArtifactEnvelope(
        artifact_id="artifact-request-1",
        correlation_id="corr-1",
        artifact_type="PolicyPlanDraft",
        payload_schema_name="PolicyPlanDraft",
        source_agent="optimization_strategy",
        target_agent="policy_dispatch",
        scope=ExecutionScope(
            session_id="session-1",
            thread_id="session-1",
            snapshot_id="snapshot-1",
            scenario_id="scenario-1",
            scenario_tags=["smoke"],
        ),
        payload={"policies": []},
    )


def _policy_plan() -> dict[str, Any]:
    return {
        "supi": "imsi-001",
        "session_id": "session-1",
        "snapshot_id": "snapshot-1",
        "policies": [],
    }


def _outcome(*, dispatch_status: str, feedback_payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        performance_metrics=json.dumps(
            {
                "policy_plan": {
                    "policies": [
                        {
                            "policy_id": "sm-1",
                            "policy_type": "SmPolicyDecision",
                            "flow_id": "flow-1",
                        }
                    ]
                },
                "dispatch_results": [
                    {
                        "status": dispatch_status,
                        "policy_id": "sm-1",
                        "policy_type": "SmPolicyDecision",
                        "flow_id": "flow-1",
                    }
                ],
                "assurance_results": (
                    [{"status": "satisfied", "policy_id": "sm-1", "flow_id": "flow-1"}]
                    if dispatch_status == "success"
                    else []
                ),
            }
        ),
        feedback_payload=feedback_payload,
        violation_details="none" if dispatch_status == "success" else "gateway rejected",
    )


def _tool_results(trace: AgentTrace) -> dict[str, Any]:
    return {message.name: message for message in trace.messages if message.role == "tool"}


def test_deterministic_execution_trace_projects_successful_execution() -> None:
    writer = _RecordingWriter()
    outcome = _outcome(dispatch_status="success", feedback_payload={})
    feedback_report = {
        "execution_status": "Success",
        "performance_metrics": outcome.performance_metrics,
        "violation_details": "none",
    }
    write_deterministic_execution_trace(
        writer=writer,
        agent_name="policy_dispatch",
        model_name="test-model",
        system_prompt="dispatch policies",
        policy_plan=_policy_plan(),
        outcome=outcome,
        feedback_report=feedback_report,
        request_envelope=_request_envelope(),
    )

    trace = writer.records[0]
    assert trace.source == "deterministic_runtime"
    assert trace.scope.session_id == "session-1"
    assert trace.targets[0].kind == "deterministic_output"
    assert set(_tool_results(trace)) == {
        "compile_policy_plan",
        "validate_policy",
        "dispatch_policy_to_pcf_request",
        "assurance_evaluate",
        "commit_policy_updates",
    }


def test_deterministic_execution_trace_retains_failed_execution_shape() -> None:
    writer = _RecordingWriter()
    outcome = _outcome(
        dispatch_status="failed",
        feedback_payload={"phase": "dispatch", "failure_scope": "qos"},
    )
    feedback_report = {
        "execution_status": "Failed",
        "performance_metrics": outcome.performance_metrics,
        "violation_details": "gateway rejected",
    }
    write_deterministic_execution_trace(
        writer=writer,
        agent_name="policy_dispatch",
        model_name="test-model",
        system_prompt="dispatch policies",
        policy_plan=_policy_plan(),
        outcome=outcome,
        feedback_report=feedback_report,
        request_envelope=_request_envelope(),
    )

    tools = _tool_results(writer.records[0])
    assert tools["dispatch_policy_to_pcf_request"].tool_status == "error"
    assert json.loads(tools["build_feedback_payload"].content) == outcome.feedback_payload
    assert "commit_policy_updates" not in tools


def test_policy_dispatch_preserves_request_correlation_on_cached_feedback() -> None:
    agent = object.__new__(PolicyDispatchAgent)
    agent.agent_name = "policy_dispatch"
    agent.model_name = "test-model"
    agent._trace_writer = _RecordingWriter()
    agent.cache = _RecordingCache()
    outcome = _outcome(dispatch_status="success", feedback_payload={})
    outcome.execution_status = "Success"
    outcome.dispatch_attempts = 1
    outcome.failure_scope = "none"
    outcome.to_dict = lambda: {
        "execution_status": "Success",
        "performance_metrics": outcome.performance_metrics,
        "violation_details": "none",
        "dispatch_attempts": 1,
    }
    agent._execution_controller = SimpleNamespace(execute=lambda _plan: outcome)
    request_envelope = _request_envelope()

    report = agent.execute_and_evaluate_from_request(
        strategy_output=_policy_plan(),
        request_envelope=request_envelope,
    )

    response = agent.cache.produced[0]
    assert response.correlation_id == request_envelope.correlation_id
    assert response.upstream_artifact_ids == [request_envelope.artifact_id]
    assert response.scope == request_envelope.scope
    assert response.payload == report.model_dump(mode="json")
