"""Canonical tracing for deterministic policy execution."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from shared.contracts import (
    AgentMessage,
    AgentTrace,
    ToolCall,
    ToolCatalog,
    ToolContract,
    TraceMetadata,
    TraceTarget,
    utc_now,
)
from shared.runtime.trace.writer import JsonlTraceWriter


_OBJECT_SCHEMA = {"type": "object"}


def _execution_tool_catalog() -> ToolCatalog:
    definitions = [
        ("compile_policy_plan", "Compile a policy plan draft.", "read"),
        ("validate_policy", "Validate one compiled policy.", "read"),
        ("dispatch_policy_to_pcf_request", "Dispatch one policy to the PCF.", "external_action"),
        ("assurance_evaluate", "Evaluate post-dispatch assurance evidence.", "read"),
        ("build_feedback_payload", "Build structured execution feedback.", "read"),
        ("commit_policy_updates", "Commit successful policy state updates.", "write"),
    ]
    return ToolCatalog.create(
        [
            ToolContract(
                name=name,
                description=description,
                input_schema=_OBJECT_SCHEMA,
                output_schema=_OBJECT_SCHEMA,
                side_effect=side_effect,
                result_format="json",
            )
            for name, description, side_effect in definitions
        ]
    )


def write_deterministic_execution_trace(
    *,
    writer: JsonlTraceWriter,
    agent_name: str,
    model_name: str,
    system_prompt: str,
    policy_plan: dict[str, Any],
    outcome: Any,
    feedback_report: dict[str, Any],
    request_envelope: Any,
    trace_metadata: dict[str, Any] | None = None,
) -> None:
    scope = request_envelope.scope.model_copy(
        update={"supi": str(policy_plan.get("supi") or "").strip() or None}
    )
    messages = [
        AgentMessage(role="system", content=system_prompt),
        AgentMessage(
            role="user",
            content=(
                "Policy plan:\n"
                f"{json.dumps(policy_plan, ensure_ascii=False)}\n\n"
                "Execute dispatch, assurance, and feedback generation."
            ),
        ),
        *_build_trace_messages(
            policy_plan=policy_plan,
            outcome=outcome,
            feedback_report=feedback_report,
        ),
        AgentMessage(
            role="assistant",
            content=json.dumps(feedback_report, ensure_ascii=False),
        ),
    ]
    metadata = dict(trace_metadata or {})
    now = utc_now()
    writer.write(
        AgentTrace(
            trace_id=f"trace-{uuid4()}",
            agent_name=agent_name,
            model_name=model_name,
            source="deterministic_runtime",
            scope=scope,
            started_at=now,
            ended_at=now,
            tool_catalog=_execution_tool_catalog(),
            response_schema_name="FeedbackReport",
            response_schema=_OBJECT_SCHEMA,
            messages=messages,
            targets=[
                TraceTarget(
                    kind="deterministic_output",
                    producer="runtime",
                    schema_name="FeedbackReport",
                    output_schema=_OBJECT_SCHEMA,
                    value=feedback_report,
                )
            ],
            status="success",
            metadata=TraceMetadata(
                path_label="policy_execution",
                prompt_name="deterministic_policy_execution",
                prompt_version="2.0.0",
                attributes={
                    key: value
                    for key, value in metadata.items()
                    if key not in {"path_label", "scenario_id", "scenario_tags"}
                },
            ),
        )
    )


def _decode_metrics_payload(outcome: Any) -> dict[str, Any]:
    metrics_text = str(outcome.performance_metrics).strip()
    metrics_payload = json.loads(metrics_text)
    if not isinstance(metrics_payload, dict):
        raise TypeError("performance_metrics must decode to a JSON object")
    return metrics_payload


def _append_tool_exchange(
    messages: list[AgentMessage],
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    status: str,
) -> None:
    call_id = f"call-{uuid4().hex}"
    messages.extend(
        [
            AgentMessage(
                role="assistant",
                tool_calls=[ToolCall(id=call_id, name=tool_name, arguments=arguments)],
            ),
            AgentMessage(
                role="tool",
                name=tool_name,
                tool_call_id=call_id,
                tool_status=status,
                content=json.dumps(result, ensure_ascii=False),
            ),
        ]
    )


def _build_trace_messages(
    *,
    policy_plan: dict[str, Any],
    outcome: Any,
    feedback_report: dict[str, Any],
) -> list[AgentMessage]:
    messages: list[AgentMessage] = []
    metrics = _decode_metrics_payload(outcome)
    failure_phase = str(outcome.feedback_payload.get("phase") or "").strip()

    compiled_plan = metrics.get("policy_plan")
    compile_success = isinstance(compiled_plan, dict) and failure_phase != "compile"
    compile_result = (
        compiled_plan
        if compile_success
        else {
            "error": str(outcome.violation_details),
            "feedback_payload": outcome.feedback_payload,
        }
    )
    _append_tool_exchange(
        messages,
        tool_name="compile_policy_plan",
        arguments={"policy_plan_draft": policy_plan},
        result=compile_result,
        status="success" if compile_success else "error",
    )
    if not compile_success:
        return messages

    policies = compiled_plan.get("policies")
    if not isinstance(policies, list) or any(not isinstance(policy, dict) for policy in policies):
        raise TypeError("compiled policy_plan.policies must be a list of objects")
    for policy in policies:
        _append_tool_exchange(
            messages,
            tool_name="validate_policy",
            arguments={"policy": policy},
            result={"status": "validated", "policy_id": policy.get("policy_id"), "policy": policy},
            status="success",
        )

    dispatch_results = metrics.get("dispatch_results")
    if not isinstance(dispatch_results, list) or any(not isinstance(item, dict) for item in dispatch_results):
        raise TypeError("dispatch_results must be a list of objects")
    for receipt in dispatch_results:
        receipt_status = "success" if str(receipt.get("status")).lower() == "success" else "error"
        _append_tool_exchange(
            messages,
            tool_name="dispatch_policy_to_pcf_request",
            arguments={
                "policy_id": receipt.get("policy_id"),
                "policy_type": receipt.get("policy_type"),
                "flow_id": receipt.get("flow_id"),
                "session_id": policy_plan.get("session_id"),
                "snapshot_id": policy_plan.get("snapshot_id"),
            },
            result=receipt,
            status=receipt_status,
        )

    assurance_results = metrics.get("assurance_results")
    if not isinstance(assurance_results, list) or any(not isinstance(item, dict) for item in assurance_results):
        raise TypeError("assurance_results must be a list of objects")
    for verdict in assurance_results:
        verdict_status = str(verdict.get("status") or "").lower()
        _append_tool_exchange(
            messages,
            tool_name="assurance_evaluate",
            arguments={
                "policy_id": verdict.get("policy_id"),
                "flow_id": verdict.get("flow_id"),
                "snapshot_id": policy_plan.get("snapshot_id"),
            },
            result=verdict,
            status="success" if verdict_status in {"satisfied", "skipped"} else "error",
        )

    if outcome.feedback_payload:
        _append_tool_exchange(
            messages,
            tool_name="build_feedback_payload",
            arguments={"failure_scope": outcome.feedback_payload.get("failure_scope")},
            result=outcome.feedback_payload,
            status="success",
        )
    if str(feedback_report.get("execution_status") or "").strip().lower() == "success":
        _append_tool_exchange(
            messages,
            tool_name="commit_policy_updates",
            arguments={
                "supi": policy_plan.get("supi"),
                "policy_ids": [policy.get("policy_id") for policy in policies],
                "session_id": policy_plan.get("session_id"),
                "snapshot_id": policy_plan.get("snapshot_id"),
            },
            result={"status": "committed"},
            status="success",
        )
    return messages


__all__ = ["write_deterministic_execution_trace"]
