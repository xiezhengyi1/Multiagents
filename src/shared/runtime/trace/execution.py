"""Projection of deterministic policy execution into the shared trace schema."""

from __future__ import annotations

import json
from typing import Any, Dict
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from ..core import AgentRuntimeContext
from ..messages import json_friendly
from .builder import build_run_tree_record, utc_now
from .writer import JsonlTraceWriter


def write_deterministic_execution_trace(
    *,
    writer: JsonlTraceWriter,
    agent_name: str,
    model_name: str,
    system_prompt: str,
    policy_plan: Dict[str, Any],
    outcome: Any,
    feedback_report: Dict[str, Any],
    request_envelope: Any,
    trace_metadata: Dict[str, Any] | None = None,
) -> None:
    """Write a faithful trace for a non-LLM policy execution lifecycle."""

    context = AgentRuntimeContext(
        agent_name=agent_name,
        session_id=request_envelope.session_id,
        snapshot_id=request_envelope.snapshot_id,
        supi=str(policy_plan.get("supi") or "").strip() or None,
        thread_id=request_envelope.session_id or request_envelope.snapshot_id or agent_name,
        allow_user_interaction=False,
        trace_metadata=dict(trace_metadata or {}),
    )
    payload = {
        "messages": [
            HumanMessage(
                content=(
                    "Policy plan:\n"
                    f"{json.dumps(policy_plan, ensure_ascii=False)}\n\n"
                    "Execute dispatch, assurance, and feedback generation."
                )
            )
        ],
        "trace_metadata": _trace_metadata_payload(trace_metadata),
    }
    result = {
        "messages": _build_trace_messages(
            policy_plan=policy_plan,
            outcome=outcome,
            feedback_report=feedback_report,
        ),
        "structured_response": feedback_report,
    }
    now = utc_now()
    writer.write(
        build_run_tree_record(
            agent_name=agent_name,
            model_name=model_name,
            system_prompt=system_prompt,
            run_id=f"run-{uuid4()}",
            payload=payload,
            context=context,
            result=result,
            status="success",
            captured_error=None,
            start_dt=now,
            end_dt=now,
        )
    )


def _trace_metadata_payload(metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    if not metadata:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError("trace_metadata must be a dict when present")
    scenario_tags = metadata.get("scenario_tags") or []
    if not isinstance(scenario_tags, list):
        raise TypeError("trace_metadata.scenario_tags must be a list")
    return {
        "scenario_id": str(metadata.get("scenario_id") or "").strip() or None,
        "scenario_tags": [str(item).strip() for item in scenario_tags if str(item).strip()],
    }


def _decode_metrics_payload(outcome: Any) -> Dict[str, Any]:
    metrics_text = str(getattr(outcome, "performance_metrics", "") or "").strip()
    metrics_payload = json.loads(metrics_text) if metrics_text else {}
    if metrics_payload and not isinstance(metrics_payload, dict):
        raise TypeError("performance_metrics must decode to a JSON object")
    return metrics_payload


def _append_trace_tool_call(
    messages: list[Any],
    *,
    tool_name: str,
    args: Dict[str, Any],
    result: Any,
    status: str = "success",
) -> None:
    tool_call_id = f"call-{uuid4().hex}"
    messages.append(
        AIMessage(
            content="",
            id=f"msg-{uuid4().hex}",
            tool_calls=[
                {
                    "id": tool_call_id,
                    "name": tool_name,
                    "args": json_friendly(args),
                }
            ],
        )
    )
    messages.append(
        {
            "type": "tool",
            "content": result if isinstance(result, str) else json.dumps(json_friendly(result), ensure_ascii=False),
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "status": status,
            "id": f"{tool_call_id}-result",
        }
    )


def _build_trace_messages(
    *,
    policy_plan: Dict[str, Any],
    outcome: Any,
    feedback_report: Dict[str, Any],
) -> list[Any]:
    messages: list[Any] = []
    metrics_payload = _decode_metrics_payload(outcome)
    failure_phase = str((getattr(outcome, "feedback_payload", {}) or {}).get("phase") or "").strip()

    compile_result = metrics_payload.get("policy_plan")
    compile_status = "error" if failure_phase == "compile" else ("success" if isinstance(compile_result, dict) else "missing")
    compile_payload: Any = compile_result
    if compile_status == "error":
        compile_payload = {
            "error": str(getattr(outcome, "violation_details", "") or "compile failed"),
            "feedback_payload": getattr(outcome, "feedback_payload", {}) or {},
        }
    elif compile_status == "missing":
        compile_payload = {"compile_status": "missing", "osa_output": policy_plan}
    _append_trace_tool_call(
        messages,
        tool_name="compile_policy_plan",
        args={"policy_plan_draft": policy_plan},
        result=compile_payload,
        status=compile_status,
    )
    if compile_status != "success":
        return messages

    policies = compile_result.get("policies") or []
    if not isinstance(policies, list):
        raise TypeError("Compiled policy list must be a list")
    for policy in policies:
        if not isinstance(policy, dict):
            raise TypeError("Each compiled policy must be a dict")
        _append_trace_tool_call(
            messages,
            tool_name="validate_policy",
            args={"policy": policy},
            result={"status": "validated", "policy_id": policy.get("policy_id"), "policy": policy},
        )

    dispatch_results = metrics_payload.get("dispatch_results") or []
    if not isinstance(dispatch_results, list):
        raise TypeError("dispatch_results must be a list")
    for receipt in dispatch_results:
        if not isinstance(receipt, dict):
            raise TypeError("Each dispatch result must be a dict")
        _append_trace_tool_call(
            messages,
            tool_name="dispatch_policy_to_pcf_request",
            args={
                "policy_id": receipt.get("policy_id"),
                "policy_type": receipt.get("policy_type"),
                "flow_id": receipt.get("flow_id"),
                "session_id": policy_plan.get("session_id"),
                "snapshot_id": policy_plan.get("snapshot_id"),
            },
            result=receipt,
            status="success" if str(receipt.get("status") or "").strip().lower() == "success" else "error",
        )

    assurance_results = metrics_payload.get("assurance_results") or []
    if not isinstance(assurance_results, list):
        raise TypeError("assurance_results must be a list")
    for verdict in assurance_results:
        if not isinstance(verdict, dict):
            raise TypeError("Each assurance result must be a dict")
        verdict_status = str(verdict.get("status") or "").strip().lower()
        _append_trace_tool_call(
            messages,
            tool_name="assurance_evaluate",
            args={
                "policy_id": verdict.get("policy_id"),
                "flow_id": verdict.get("flow_id"),
                "snapshot_id": policy_plan.get("snapshot_id"),
            },
            result=verdict,
            status="success" if verdict_status in {"satisfied", "skipped"} else "error",
        )

    feedback_payload = getattr(outcome, "feedback_payload", {}) or {}
    if not isinstance(feedback_payload, dict):
        raise TypeError("feedback_payload must be a dict")
    if feedback_payload:
        _append_trace_tool_call(
            messages,
            tool_name="build_feedback_payload",
            args={"failure_scope": feedback_payload.get("failure_scope") or ""},
            result=feedback_payload,
            status="success",
        )

    if str(feedback_report.get("execution_status") or "").strip().lower() == "success":
        _append_trace_tool_call(
            messages,
            tool_name="commit_policy_updates",
            args={
                "supi": policy_plan.get("supi"),
                "policy_ids": [policy.get("policy_id") for policy in policies],
                "session_id": policy_plan.get("session_id"),
                "snapshot_id": policy_plan.get("snapshot_id"),
            },
            result={"status": "committed", "execution_status": feedback_report.get("execution_status")},
            status="success",
        )

    return messages


__all__ = ["write_deterministic_execution_trace"]
