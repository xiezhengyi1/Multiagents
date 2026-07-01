from __future__ import annotations

from typing import Any, Dict, List

from ...domain.collaboration import PlanningContext, PlanningRequest
from ...domain.control_plane import (
    ControlDomain,
    JointOptimizationRequest,
    ObjectiveProfile,
    OptimizationProblemConfig,
    OptimizationTemplate,
)
from ...domain.policy_plan import OperationIntent
from ..projectors import ProjectorRegistry
from .normalizer import normalize_app_id


class EvidenceFormatter:
    """Unified evidence formatting for agent, optimizer, and LLM consumers."""

    @classmethod
    def for_iea(
        cls,
        *,
        user_input: str,
        supi: str,
        main_directives: Dict[str, Any],
        catalog_payload: Dict[str, Any],
        semantic_candidates: List[Dict[str, Any]],
        am_context_payload: Dict[str, Any] | None = None,
        am_policy_candidates: List[Dict[str, Any]] | None = None,
    ) -> Any:
        # Keep the mature grounding builder as the implementation source while
        # moving the public construction boundary into context.evidence.
        from .grounding import IntentEvidenceBuilder

        return IntentEvidenceBuilder().build_intent_evidence(
            user_input=user_input,
            supi=supi,
            main_directives=main_directives,
            catalog_payload=catalog_payload,
            semantic_candidates=semantic_candidates,
            am_context_payload=am_context_payload,
            am_policy_candidates=am_policy_candidates,
        )

    @classmethod
    def for_osa(
        cls,
        *,
        operation_intent: OperationIntent,
        planning_context: PlanningContext,
    ) -> Dict[str, Any]:
        flows: List[Dict[str, Any]] = []
        for flow in operation_intent.flows:
            flows.append(
                {
                    "flow_id": str(flow.flow_id or "").strip(),
                    "app_id": normalize_app_id(flow.app_id),
                    "name": str(flow.name or "").strip(),
                    "priority": flow.priority,
                    "service_type_id": flow.service_type_id,
                    "current_slice_snssai": str(flow.current_slice_snssai or "").strip() or None,
                }
            )
        qos_objectives = [
            objective.model_dump(mode="json")
            for objective in operation_intent.qos_target_envelopes
        ]
        return {
            "requested_domains": list(planning_context.active_domains or []),
            "main_retry_scope": str(planning_context.main_retry_scope or "").strip(),
            "objective_profile": dict(planning_context.objective_profile or {}),
            "required_evidence": list(planning_context.required_evidence or []),
            "forbidden_assumptions": list(planning_context.forbidden_assumptions or []),
            "revision_requests": list(planning_context.revision_requests or []),
            "unified_constraints": dict(planning_context.unified_constraints or {}),
            "flows": flows,
            "qos_target_envelopes": qos_objectives,
        }

    @classmethod
    def for_optimizer(
        cls,
        planning_request: PlanningRequest,
        *,
        profile_name: str | None = None,
        template_name: str | None = None,
        qos_relaxation_ratio: float | None = None,
        slice_kpi_source: str | None = None,
    ) -> JointOptimizationRequest:
        from ...integrations.storage import get_snapshot_data_by_id, get_ue_context_by_supi

        operation_intent = planning_request.operation_intent
        snapshot_id = str(planning_request.context.snapshot_id or "").strip()
        if not snapshot_id:
            raise ValueError("optimizer request requires a bound snapshot_id")
        snapshot = get_snapshot_data_by_id(snapshot_id) or {}
        if not snapshot:
            raise LookupError(f"optimizer request snapshot not found: snapshot_id={snapshot_id}")
        target_supis = _collect_target_supis(planning_request)

        requested_domains: list[ControlDomain] = []
        for item in planning_request.context.active_domains or []:
            normalized = str(item or "").strip().lower()
            if normalized in {"both", "all"}:
                requested_domains = [ControlDomain.QOS, ControlDomain.MOBILITY]
                break
            requested_domains.append(ControlDomain(normalized))
        if not requested_domains:
            raise ValueError("optimizer request requires non-empty active_domains from Main/IEA")

        ue_contexts = {
            supi: (get_ue_context_by_supi(supi, snapshot_id=snapshot_id) or {})
            for supi in target_supis
        }

        objective_profile_payload = dict(planning_request.context.objective_profile or {})
        if not objective_profile_payload:
            raise ValueError("optimizer request requires an explicit objective_profile")
        if profile_name is not None:
            normalized_profile = str(profile_name or "").strip().lower()
            if not normalized_profile:
                raise ValueError("profile_name must not be empty when provided")
            objective_profile_payload["profile_name"] = normalized_profile

        problem_config = OptimizationProblemConfig()
        if template_name is not None:
            normalized_template = str(template_name or "").strip().lower()
            if not normalized_template:
                raise ValueError("template_name must not be empty when provided")
            problem_config = OptimizationProblemConfig(template=OptimizationTemplate(normalized_template))
        if qos_relaxation_ratio is not None:
            problem_config.qos_relaxation_ratio = float(qos_relaxation_ratio)
        if slice_kpi_source is not None:
            normalized_source = str(slice_kpi_source or "").strip().lower()
            if normalized_source not in {"qos", "telemetry"}:
                raise ValueError("slice_kpi_source must be either 'qos' or 'telemetry'")
            problem_config.slice_kpi_source = normalized_source

        return JointOptimizationRequest(
            session_id=planning_request.context.session_id,
            snapshot_id=snapshot_id,
            target_ues=target_supis,
            requested_domains=requested_domains,
            operation_intent=operation_intent.model_dump(mode="json"),
            traffic_state={
                "apps": snapshot.get("apps", []),
                "slices": snapshot.get("slices", []),
                "nodes": snapshot.get("nodes", []),
            },
            resource_state={
                "slices": snapshot.get("slices", []),
                "nodes": snapshot.get("nodes", []),
            },
            mobility_state={
                supi: ctx.get("accessMobilityContext", {})
                for supi, ctx in ue_contexts.items()
                if ctx
            },
            policy_state={supi: ctx for supi, ctx in ue_contexts.items() if ctx},
            objective_profile=ObjectiveProfile.model_validate(objective_profile_payload),
            problem_config=problem_config,
        )

    @classmethod
    def for_llm(cls, *, model: Any, role: str = "") -> dict[str, Any]:
        projector = ProjectorRegistry.for_instance(model)
        return projector.project(model)


def _collect_target_supis(planning_request: PlanningRequest) -> list[str]:
    operation_intent = planning_request.operation_intent
    target_supis: list[str] = []

    def _append(candidate: str) -> None:
        supi = str(candidate or "").strip()
        if supi and supi not in target_supis:
            target_supis.append(supi)

    _append(operation_intent.supi)
    for flow in operation_intent.flows or []:
        _append(flow.supi)
    return target_supis
