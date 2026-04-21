from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable

from agents.BaseAgent import coerce_structured_response

from .prompts import OSA_SYSTEM_PROMPT
from .response_models import OsaAdvisorOutput
from .tools import _summarize_optimizer_result, build_request_tools


@dataclass
class AdvisorInvocation:
    advisor_output: OsaAdvisorOutput
    raw_result: dict[str, Any]
    trace_agent: Any
    trace_payload: dict[str, Any]
    runtime_context: Any

    def write_final_trace(
        self,
        *,
        status: str,
        error: str | None = None,
        compiler_output: Any = None,
    ) -> None:
        payload = copy.deepcopy(self.trace_payload)
        metadata = dict(payload.get("trace_metadata") or {})
        metadata["advisor_decision"] = self.advisor_output.model_dump(mode="json")
        metadata["compiler_output"] = compiler_output
        payload["trace_metadata"] = metadata
        self.trace_agent.write_trace(
            payload=payload,
            context=self.runtime_context,
            result=self.raw_result,
            status=status,
            error=error,
            structured_response_override=compiler_output,
        )


class OptimizationStrategyAdvisor:
    def __init__(self, owner: Any, compiler: Any) -> None:
        self.owner = owner
        self.compiler = compiler

    def advise(
        self,
        *,
        planning_request: Any,
        normalized_user_intent: Dict[str, Any],
        coordination_context: Dict[str, Any],
        optimizer_preview: Any,
        planning_evidence: Dict[str, Any],
    ) -> AdvisorInvocation:
        runtime_context = self.owner.build_runtime_context(
            agent_name=self.owner.agent_name,
            session_id=planning_request.context.session_id,
            snapshot_id=planning_request.context.snapshot_id,
            supi=planning_request.operation_intent.supi,
            thread_id=planning_request.context.session_id,
        )
        preview_payload = _summarize_optimizer_result(optimizer_preview)
        request_tools = build_request_tools(planning_request)
        all_tools = [*self.owner._BASE_TOOLS, *request_tools]
        advisor_agent = self.owner.create_json_agent(
            tools=all_tools,
            system_prompt=OSA_SYSTEM_PROMPT,
            response_model=OsaAdvisorOutput,
            max_iterations=14,
        )

        messages = [
            {
                "role": "user",
                "content": build_advisor_user_prompt(
                    normalized_user_intent=normalized_user_intent,
                    coordination_context=coordination_context,
                    planning_evidence=planning_evidence,
                    optimizer_preview_summary=preview_payload,
                ),
            }
        ]
        self.owner._pending_invoke_messages = messages
        base_trace_metadata = getattr(self.owner, "_pending_trace_metadata", {}) or {}
        invoke_payload = {
            "messages": messages,
            "trace_write_mode": "manual",
            "trace_metadata": {
                **base_trace_metadata,
                "path_label": "strategy_advisor",
            },
        }
        try:
            result = advisor_agent.invoke(invoke_payload, context=runtime_context)
        except Exception as exc:
            repair_messages = [
                {
                    "role": "user",
                        "content": (
                            f"{messages[-1]['content']}\n\n"
                            "Your previous response was not valid raw JSON for OsaAdvisorOutput.\n"
                            f"Parser error: {exc}\n\n"
                            "Return only one JSON object. Do not use markdown fences.\n"
                            "Do not output nested qos objects, target_snssai objects inside sm_policies, full request/policy payloads, supi, ue_ambr, or pras."
                        ),
                }
            ]
            self.owner._pending_invoke_messages = repair_messages
            invoke_payload = {
                "messages": repair_messages,
                "trace_write_mode": "manual",
                "trace_metadata": {
                    **base_trace_metadata,
                    "path_label": "strategy_advisor",
                },
            }
            result = advisor_agent.invoke(invoke_payload, context=runtime_context)

        advisor_output = coerce_structured_response(
            result,
            OsaAdvisorOutput,
            error_message="OSA advisor returned no structured_response",
        )
        return AdvisorInvocation(
            advisor_output=advisor_output,
            raw_result=result,
            trace_agent=advisor_agent,
            trace_payload=invoke_payload,
            runtime_context=runtime_context,
        )


def _normalize_domains(raw_domains: Iterable[Any]) -> list[str]:
    normalized: list[str] = []
    for item in raw_domains:
        value = str(item or "").strip().lower()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _required_tool_instructions(active_domains: list[str], planning_evidence: Dict[str, Any]) -> str:
    instructions: list[str] = []
    if "qos" in active_domains:
        if bool(planning_evidence.get("preview_qos_plan_present")):
            instructions.append(
                "- QoS is active: the initial optimizer preview already provides grounded QoS runtime evidence. Reuse it directly unless you need fresher comparison evidence or the preview leaves required fields unsupported."
            )
        else:
            instructions.append(
                "- QoS is active but the initial optimizer preview does not provide grounded QoS runtime evidence. Call `preview_optimizer` or `fetch_network_status` before final JSON."
            )
    if "mobility" in active_domains:
        if bool(planning_evidence.get("preview_mobility_plan_present")):
            instructions.append(
                "- Mobility is active: the initial optimizer preview already provides grounded mobility context. Reuse it directly unless required fields remain unsupported or you need fresher UE-policy evidence."
            )
        else:
            instructions.append(
                "- Mobility is active but the initial optimizer preview does not provide grounded mobility context. Call `inspect_ue_policies` before final JSON."
            )
    if not instructions:
        instructions.append("- Call a grounding tool before final JSON if any policy field depends on runtime evidence.")
    return "\n".join(instructions)


def _output_schema_contract(active_domains: list[str]) -> str:
    lines = [
        "Return exact schema fields only.",
        "- Top-level keys allowed: `rationale`, `sm_policies`, `am_policy`, `ursp_policies`, `planning_metadata`.",
        "- Never output full `request`, `policy`, `qos`, `target_snssai`, `supi`, `ue_ambr`, or `pras` objects.",
    ]
    if "qos" in active_domains:
        lines.append(
            "- Each `sm_policies` item must use only: `flow_id`, `app_id`, `priority`, `target_latency_ms`, `packet_error_rate`, `max_br_ul_mbps`, `max_br_dl_mbps`, optional `gbr_ul_mbps`, `gbr_dl_mbps`, `target_jitter_ms`, `flow_description`."
        )
    if "mobility" in active_domains:
        lines.append(
            "- `am_policy` must use only: `triggers`, `rfsp`, `allowed_snssais`, `target_snssais`, optional `ue_ambr_ul_mbps`, `ue_ambr_dl_mbps`, `serv_area_res`, `rationale`."
        )
    lines.append(
        "- `allowed_snssais` and `target_snssais` must be lists of `{ \"sst\": int, \"sd\": \"6-hex\" }` objects."
    )
    return "\n".join(lines)


def build_advisor_user_prompt(
    *,
    normalized_user_intent: Dict[str, Any],
    coordination_context: Dict[str, Any],
    planning_evidence: Dict[str, Any],
    optimizer_preview_summary: Dict[str, Any],
) -> str:
    active_domains = _normalize_domains(coordination_context.get("active_domains") or planning_evidence.get("requested_domains") or [])
    return (
        "Structured operation intent:\n"
        f"{json.dumps(normalized_user_intent, ensure_ascii=False)}\n\n"
        "Planning context:\n"
        f"{json.dumps(coordination_context, ensure_ascii=False)}\n\n"
        "Planning evidence:\n"
        f"{json.dumps(planning_evidence, ensure_ascii=False)}\n\n"
        "Initial optimizer preview summary:\n"
        f"{json.dumps(optimizer_preview_summary, ensure_ascii=False)}\n\n"
        "Mandatory tool-use rules for this round:\n"
        f"{_required_tool_instructions(active_domains, planning_evidence)}\n\n"
        "Output contract:\n"
        f"{_output_schema_contract(active_domains)}\n\n"
        "Reasoning requirement:\n"
        "- Infer the final minimal fields from the provided evidence and tool results.\n"
        "- Do not imitate a canned template or restate the optimizer preview as nested payloads.\n"
        "- When a field is not grounded by evidence, omit it if optional; if required, collect more evidence first.\n\n"
        "Return one OsaAdvisorOutput JSON object only."
    )


__all__ = ["AdvisorInvocation", "OptimizationStrategyAdvisor", "build_advisor_user_prompt"]
