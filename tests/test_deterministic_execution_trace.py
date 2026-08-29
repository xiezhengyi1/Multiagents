from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from control_runtime.agents.dispatch.agent import PolicyDispatchAgent
from shared.runtime import ArtifactEnvelope
from shared.runtime.trace.execution import write_deterministic_execution_trace


class _RecordingWriter:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)


class _RecordingCache:
    def __init__(self) -> None:
        self.produced: list[ArtifactEnvelope] = []

    def cache_produced(self, envelope: ArtifactEnvelope) -> None:
        self.produced.append(envelope)


def _request_envelope() -> SimpleNamespace:
    return SimpleNamespace(
        artifact_id="artifact-request-1",
        correlation_id="corr-1",
        source_agent="optimization_strategy",
        session_id="session-1",
        snapshot_id="snapshot-1",
    )


def _policy_plan() -> dict[str, Any]:
    return {
        "supi": "imsi-001",
        "session_id": "session-1",
        "snapshot_id": "snapshot-1",
        "policies": [],
    }


def _tool_runs(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        tool["name"]: tool
        for llm_run in record["child_runs"]
        for tool in llm_run["child_runs"]
    }


def test_deterministic_execution_trace_projects_successful_execution() -> None:
    writer = _RecordingWriter()
    outcome = SimpleNamespace(
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
                        "status": "success",
                        "policy_id": "sm-1",
                        "policy_type": "SmPolicyDecision",
                        "flow_id": "flow-1",
                    }
                ],
                "assurance_results": [
                    {"status": "satisfied", "policy_id": "sm-1", "flow_id": "flow-1"}
                ],
            }
        ),
        feedback_payload={},
        violation_details="none",
    )
    feedback_report = {
        "execution_status": "Success",
        "intent_status": "satisfied",
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
        trace_metadata={"scenario_id": "scenario-1", "scenario_tags": ["smoke"]},
    )

    assert len(writer.records) == 1
    record = writer.records[0]
    assert record["status"] == "success"
    assert record["metadata"]["session_id"] == "session-1"
    assert record["metadata"]["snapshot_id"] == "snapshot-1"
    assert record["metadata"]["scenario_id"] == "scenario-1"
    assert record["outputs"]["structured_response"] == feedback_report
    assert set(_tool_runs(record)) == {
        "compile_policy_plan",
        "validate_policy",
        "dispatch_policy_to_pcf_request",
        "assurance_evaluate",
        "commit_policy_updates",
    }


def test_deterministic_execution_trace_retains_failed_execution_shape() -> None:
    writer = _RecordingWriter()
    outcome = SimpleNamespace(
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
                        "status": "failed",
                        "policy_id": "sm-1",
                        "policy_type": "SmPolicyDecision",
                        "flow_id": "flow-1",
                        "error": "gateway rejected",
                    }
                ],
                "assurance_results": [],
            }
        ),
        feedback_payload={"phase": "dispatch", "failure_scope": "qos"},
        violation_details="gateway rejected",
    )
    feedback_report = {
        "execution_status": "Failed",
        "intent_status": "unsatisfied",
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

    record = writer.records[0]
    tools = _tool_runs(record)
    assert record["status"] == "success"
    assert tools["dispatch_policy_to_pcf_request"]["outputs"]["result"]["status"] == "error"
    assert tools["build_feedback_payload"]["outputs"]["result"]["content"] == json.dumps(
        outcome.feedback_payload,
        ensure_ascii=False,
    )
    assert "commit_policy_updates" not in tools


def test_policy_dispatch_preserves_request_correlation_on_cached_feedback() -> None:
    agent = object.__new__(PolicyDispatchAgent)
    agent.agent_name = "policy_dispatch"
    agent.model_name = "test-model"
    agent._trace_writer = _RecordingWriter()
    agent.cache = _RecordingCache()
    outcome = SimpleNamespace(
        execution_status="Success",
        performance_metrics=json.dumps(
            {
                "policy_plan": {"policies": []},
                "dispatch_results": [],
                "assurance_results": [],
            }
        ),
        violation_details="none",
        feedback_payload={},
        dispatch_attempts=1,
        failure_scope="none",
        to_dict=lambda: {
            "execution_status": "Success",
            "performance_metrics": json.dumps(
                {"policy_plan": {"policies": []}, "dispatch_results": [], "assurance_results": []}
            ),
            "violation_details": "none",
            "dispatch_attempts": 1,
        },
    )
    agent._execution_controller = SimpleNamespace(execute=lambda _plan: outcome)
    request_envelope = ArtifactEnvelope(
        artifact_id="artifact-request-1",
        correlation_id="corr-1",
        artifact_type="PolicyPlanDraft",
        source_agent="optimization_strategy",
        target_agent="policy_dispatch",
        session_id="session-1",
        snapshot_id="snapshot-1",
    )

    report = agent.execute_and_evaluate_from_request(
        strategy_output=_policy_plan(),
        request_envelope=request_envelope,
    )

    assert report.execution_status == "Success"
    assert len(agent.cache.produced) == 1
    response = agent.cache.produced[0]
    assert response.correlation_id == request_envelope.correlation_id
    assert response.upstream_artifact_ids == [request_envelope.artifact_id]
    assert response.payload == report.model_dump(mode="json")
