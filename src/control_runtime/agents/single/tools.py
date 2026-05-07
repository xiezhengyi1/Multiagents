from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from langchain.tools import ToolRuntime

from shared.runtime import AgentRuntimeContext
from shared.tools.wrapper_think import tool_with_reason

from ...domain.collaboration import PlanningContext, PlanningRequest
from ...domain.policy_plan import FlowSelector, OperationIntent, QosTargetEnvelope
from ...integrations.scenario.network_status import get_network_status_summary
from ...integrations.storage import get_ue_context_by_supi, get_ue_flow_catalog_by_supi
from ..planning.request_builder import build_joint_optimizer_request
from ..planning.tools import _serialize_optimizer_result, _summarize_optimizer_result


def _normalize_domains(requested_domains: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    for item in requested_domains or []:
        text = str(item or "").strip().lower()
        if text and text not in normalized:
            normalized.append(text)
    if not normalized:
        raise ValueError("requested_domains must not be empty")
    return normalized


def _resolve_flow_catalog_rows(supi: str, flow_ids: List[str], snapshot_id: str) -> List[Dict[str, Any]]:
    catalog = get_ue_flow_catalog_by_supi(supi, snapshot_id=snapshot_id) or {}
    flow_catalog = catalog.get("flow_catalog") if isinstance(catalog, dict) else []
    normalized_flow_ids = [str(item or "").strip() for item in flow_ids if str(item or "").strip()]
    if not normalized_flow_ids:
        raise ValueError("flow_ids must not be empty")
    rows = [
        item
        for item in flow_catalog or []
        if isinstance(item, dict) and str(item.get("flow_id") or "").strip() in normalized_flow_ids
    ]
    found_ids = {str(item.get("flow_id") or "").strip() for item in rows}
    missing = [item for item in normalized_flow_ids if item not in found_ids]
    if missing:
        raise LookupError(f"flow_ids not found in UE flow catalog: {missing}")
    return rows


def _runtime_snapshot_id(runtime: ToolRuntime[AgentRuntimeContext]) -> str:
    snapshot_id = str(runtime.context.snapshot_id or "").strip() if runtime is not None and runtime.context is not None else ""
    if not snapshot_id:
        raise ValueError("agent tool requires a bound snapshot_id")
    return snapshot_id


def _flow_selector_from_catalog(flow: Dict[str, Any], supi: str) -> FlowSelector:
    service = flow.get("service") if isinstance(flow.get("service"), dict) else {}
    sla = flow.get("sla") if isinstance(flow.get("sla"), dict) else {}
    traffic = flow.get("traffic") if isinstance(flow.get("traffic"), dict) else {}
    return FlowSelector(
        supi=supi,
        app_id=str(flow.get("app_id") or "").strip(),
        app_name=str(flow.get("app_name") or "").strip() or None,
        flow_id=str(flow.get("flow_id") or "").strip() or None,
        target_type="flow",
        name=str(flow.get("flow_name") or flow.get("flow_id") or "").strip(),
        service_type=str(service.get("service_type") or "").strip() or None,
        service_type_id=service.get("service_type_id"),
        bw_ul=sla.get("bandwidth_ul"),
        bw_dl=sla.get("bandwidth_dl"),
        gbr_ul=sla.get("guaranteed_bandwidth_ul"),
        gbr_dl=sla.get("guaranteed_bandwidth_dl"),
        lat=sla.get("latency"),
        loss_req=sla.get("loss_rate"),
        jitter_req=sla.get("jitter"),
        priority=sla.get("priority"),
        description=str(flow.get("flow_name") or "").strip() or None,
        five_tuple=list(traffic.get("five_tuple")) if isinstance(traffic.get("five_tuple"), (list, tuple)) else None,
        resolution_status="resolved",
    )


def _profile_request_signals(objective_profile: str) -> Dict[str, bool]:
    lowered = str(objective_profile or "").strip().lower()
    return {
        "latency": any(token in lowered for token in ("latency", "low_latency", "delay", "fast")),
        "jitter": any(token in lowered for token in ("stability", "stable", "jitter")),
        "reliability": any(token in lowered for token in ("stability", "stable", "reliability", "control")),
        "throughput": any(token in lowered for token in ("throughput", "bandwidth", "capacity")),
        "uplink_only": "uplink" in lowered,
        "downlink_only": "downlink" in lowered,
        "strong_control": any(token in lowered for token in ("stability", "control", "medical", "industrial")),
    }


def _derive_strictest_latency(baseline: Optional[float], request_signals: Dict[str, bool]) -> Optional[float]:
    if baseline is None:
        return None
    factor = 1.0
    if request_signals.get("latency"):
        factor = 0.9
    if request_signals.get("strong_control"):
        factor = min(factor, 0.85)
    return round(max(baseline * factor, 1.0), 3)


def _derive_strictest_jitter(baseline: Optional[float], request_signals: Dict[str, bool]) -> Optional[float]:
    if baseline is None:
        return None
    factor = 1.0
    if request_signals.get("jitter") or request_signals.get("reliability"):
        factor = 0.9
    if request_signals.get("strong_control"):
        factor = min(factor, 0.85)
    return round(max(baseline * factor, 0.0), 3)


def _derive_strictest_loss(baseline: Optional[float], request_signals: Dict[str, bool]) -> Optional[float]:
    if baseline is None:
        return None
    factor = 1.0
    if request_signals.get("reliability") or request_signals.get("jitter"):
        factor = 0.9
    if request_signals.get("strong_control"):
        factor = min(factor, 0.85)
    return round(max(baseline * factor, 0.0), 6)


def _derive_strictest_bandwidth(
    baseline: Optional[float],
    request_signals: Dict[str, bool],
    *,
    direction: str,
) -> Optional[float]:
    if baseline is None:
        return None
    if not request_signals.get("throughput"):
        return round(max(baseline, 0.0), 3)
    if request_signals.get("uplink_only") and direction == "dl":
        return round(max(baseline, 0.0), 3)
    if request_signals.get("downlink_only") and direction == "ul":
        return round(max(baseline, 0.0), 3)
    factor = 1.1
    if request_signals.get("strong_control"):
        factor = 1.05
    return round(max(baseline * factor, 0.0), 3)


def _build_qos_target_envelopes(flows: List[FlowSelector], objective_profile: str) -> List[QosTargetEnvelope]:
    request_signals = _profile_request_signals(objective_profile)
    envelopes: List[QosTargetEnvelope] = []
    for flow in flows:
        flow_id = str(flow.flow_id or "").strip()
        if not flow_id:
            continue
        envelopes.append(
            QosTargetEnvelope(
                flow_id=flow_id,
                app_id=str(flow.app_id or "").strip(),
                flow_name=str(flow.name or flow_id).strip(),
                baseline_priority=flow.priority,
                baseline_latency_ms=flow.lat,
                baseline_jitter_ms=flow.jitter_req,
                baseline_packet_error_rate=flow.loss_req,
                baseline_max_br_ul_mbps=flow.bw_ul,
                baseline_max_br_dl_mbps=flow.bw_dl,
                baseline_gbr_ul_mbps=flow.gbr_ul,
                baseline_gbr_dl_mbps=flow.gbr_dl,
                strictest_priority=flow.priority,
                strictest_latency_ms=_derive_strictest_latency(flow.lat, request_signals),
                strictest_jitter_ms=_derive_strictest_jitter(flow.jitter_req, request_signals),
                strictest_packet_error_rate=_derive_strictest_loss(flow.loss_req, request_signals),
                strictest_max_br_ul_mbps=_derive_strictest_bandwidth(flow.bw_ul, request_signals, direction="ul"),
                strictest_max_br_dl_mbps=_derive_strictest_bandwidth(flow.bw_dl, request_signals, direction="dl"),
                strictest_gbr_ul_mbps=_derive_strictest_bandwidth(flow.gbr_ul, request_signals, direction="ul"),
                strictest_gbr_dl_mbps=_derive_strictest_bandwidth(flow.gbr_dl, request_signals, direction="dl"),
                rationale=[f"grounded_from_flow:{flow_id}", f"objective_profile:{str(objective_profile or 'balanced').strip() or 'balanced'}"],
            )
        )
    return envelopes


def _build_runtime_planning_request(
    *,
    supi: str,
    flow_ids: List[str],
    requested_domains: List[str],
    objective_profile: str,
    session_id: str,
    snapshot_id: str,
) -> PlanningRequest:
    normalized_supi = str(supi or "").strip()
    if not normalized_supi:
        raise ValueError("supi must not be empty")
    normalized_domains = _normalize_domains(requested_domains)
    flow_rows = _resolve_flow_catalog_rows(normalized_supi, flow_ids, snapshot_id)
    flows = [_flow_selector_from_catalog(item, normalized_supi) for item in flow_rows]
    operation_intent = OperationIntent(
        session_id=session_id,
        snapshot_id=snapshot_id,
        supi=normalized_supi,
        app_id=str(flows[0].app_id or "").strip() if flows else "",
        operation_type="modify",
        raw_input="",
        raw_intent_summary="",
        resolution_status="resolved",
        requested_domains=normalized_domains,
        domain_evidence={},
        objective_profile_hint=str(objective_profile or "").strip(),
        flows=flows,
        qos_target_envelopes=_build_qos_target_envelopes(flows, objective_profile),
    )
    return PlanningRequest(
        operation_intent=operation_intent,
        context=PlanningContext(
            round_index=1,
            session_id=session_id,
            snapshot_id=snapshot_id,
            snapshot_metadata={},
            active_domains=normalized_domains,
            main_round_strategy="initial_grounding",
            objective_profile={"profile_name": str(objective_profile or "balanced").strip() or "balanced"},
            required_evidence=["qos_runtime_evidence"],
        ),
    )


def build_single_agent_tools(
    *,
    rag_enabled: bool,
    requested_domains: Optional[List[str]] = None,
    allow_knowledge_tools: bool = False,
) -> List[Any]:
    from ...integrations.pcf import (
        get_am_policy_context,
        get_sm_ue_context,
        get_sm_ue_flow_catalog,
        search_am_policy_targets,
        search_sm_flow_targets,
    )
    from ...integrations.optimizer import run_joint_control_optimizer as run_optimizer

    @tool_with_reason
    def preview_qos_optimizer(
        supi: str,
        flow_ids: List[str],
        requested_domains: List[str],
        objective_profile: Literal["balanced", "latency", "throughput", "stability"] = "balanced",
        runtime: ToolRuntime[AgentRuntimeContext] = None,
    ) -> str:
        """Run the optimizer after single-agent grounding and return summary plus full result."""
        runtime_context = runtime.context if runtime is not None else None
        planning_request = _build_runtime_planning_request(
            supi=supi,
            flow_ids=flow_ids,
            requested_domains=requested_domains,
            objective_profile=objective_profile,
            session_id=str(runtime_context.session_id or "") if runtime_context is not None else "",
            snapshot_id=_runtime_snapshot_id(runtime),
        )
        result = run_optimizer(
            build_joint_optimizer_request(
                planning_request,
                profile_name=str(objective_profile or "balanced").strip().lower(),
                template_name="joint_balanced",
                qos_relaxation_ratio=0.2,
                slice_kpi_source="qos",
            )
        )
        full_payload = _serialize_optimizer_result(result)
        return json.dumps(
            {
                "summary": _summarize_optimizer_result(full_payload),
                "result": full_payload,
            },
            ensure_ascii=False,
        )

    @tool_with_reason
    def fetch_qos_network_status(
        service_type_id: Optional[int] = None,
        runtime: ToolRuntime[AgentRuntimeContext] = None,
    ) -> str:
        """Fetch QoS-domain network slice utilization and capacity summary."""
        return get_network_status_summary(
            flow_type_id=service_type_id,
            snapshot_id=_runtime_snapshot_id(runtime),
        )

    @tool_with_reason
    def inspect_mobility_ue_policies(
        supi: str,
        runtime: ToolRuntime[AgentRuntimeContext] = None,
    ) -> str:
        """Inspect current UE mobility policy state for a grounded SUPI."""
        normalized_supi = str(supi or "").strip()
        if not normalized_supi:
            raise ValueError("inspect_mobility_ue_policies requires supi")
        ue_ctx = get_ue_context_by_supi(
            normalized_supi,
            snapshot_id=(runtime.context.snapshot_id if runtime is not None and runtime.context is not None else ""),
        )
        if not ue_ctx:
            raise LookupError(f"No UE context found for {normalized_supi}")
        trimmed: Dict[str, Any] = {}
        for key in (
            "supi",
            "accessMobilityContext",
            "amPolicyContext",
            "mobilitySummary",
            "servingNfContext",
        ):
            if key in ue_ctx:
                trimmed[key] = ue_ctx[key]
        if not trimmed:
            raise RuntimeError(f"UE context for {normalized_supi} contains no policy-relevant mobility fields")
        return json.dumps(trimmed, ensure_ascii=False)

    normalized_domains = _normalize_domains(requested_domains or ["qos", "mobility"])
    tools: List[Any] = []
    if "qos" in normalized_domains:
        tools.extend(
            [
                search_sm_flow_targets,
                get_sm_ue_context,
                get_sm_ue_flow_catalog,
                preview_qos_optimizer,
                fetch_qos_network_status,
            ]
        )
    if "mobility" in normalized_domains:
        tools.extend(
            [
                get_am_policy_context,
                search_am_policy_targets,
                inspect_mobility_ue_policies,
            ]
        )
    if rag_enabled and allow_knowledge_tools:
        from knowledge_runtime.retrieval.raw import get_knowledge_by_key, search_semantic_knowledge

        tools.extend([search_semantic_knowledge, get_knowledge_by_key])
    return tools


__all__ = ["build_single_agent_tools"]
