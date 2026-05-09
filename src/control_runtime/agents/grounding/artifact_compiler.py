from __future__ import annotations

from typing import Dict, List, Optional

from ...domain.policy_plan import FlowSelector, GroundingEvidenceBundle, OperationIntent
from .common import (
    classify_domain_resolution,
    flow_id_is_grounded,
    normalize_domain_evidence,
    normalize_requested_domains,
    uses_am_grounding,
    uses_sm_grounding,
)
from .control_semantics_grounder import ControlSemanticsGrounder
from .contracts import FlowCandidateEvidence, IntentAdvisorDecision, IntentEvidence
from .qos_envelope_builder import QosEnvelopeBuilder


class OperationIntentCompiler:
    def __init__(self) -> None:
        self.control_semantics_grounder = ControlSemanticsGrounder()
        self.qos_envelope_builder = QosEnvelopeBuilder()

    def compile_operation_intent(
        self,
        *,
        evidence: IntentEvidence,
        advisor_decision: Optional[IntentAdvisorDecision],
        user_input: str,
        session_id: str,
        snapshot_id: str,
        main_directives: Optional[Dict[str, Any]] = None,
    ) -> OperationIntent:
        if advisor_decision is None:
            raise ValueError("advisor_decision is required to compile OperationIntent")
        decision = advisor_decision
        directives = main_directives or {}

        catalog_payload = dict(evidence.catalog_payload or {})
        app_catalog = catalog_payload.get("app_catalog") or []
        flow_catalog = catalog_payload.get("flow_catalog") or []
        semantic_candidates = list(evidence.semantic_candidates or [])

        selected_flow = self._resolve_selected_flow(
            decision=decision,
            flow_catalog=flow_catalog,
            semantic_candidates=semantic_candidates,
        )
        selected_app_id = self._resolve_selected_app_id(
            decision=decision,
            selected_flow=selected_flow,
        )
        app_name = self._resolve_selected_app_name(app_catalog=app_catalog, selected_app_id=selected_app_id)
        flows = self._build_compiled_flows(
            evidence=evidence,
            decision=decision,
            selected_app_id=selected_app_id,
            flow_catalog=flow_catalog,
            semantic_candidates=semantic_candidates,
        )
        resolution_status = self._resolve_operation_status(
            requested_domains=evidence.requested_domains,
            candidate_flows=evidence.candidate_flows,
            flows=flows,
        )
        grounded_requested_domains = normalize_requested_domains(
            decision.grounded_requested_domains or evidence.requested_domains
        )
        domain_resolution, domain_revision_needed = classify_domain_resolution(
            main_requested_domains=evidence.main_requested_domains,
            grounded_requested_domains=grounded_requested_domains,
            decision=decision,
        )
        resolved_mobility_intent = self._resolve_mobility_intent(
            decision=decision,
        )
        qos_target_envelopes = self.qos_envelope_builder.build(
            flows=flows,
        )
        control_semantics = self.control_semantics_grounder.ground(
            directives=directives,
            flows=flows,
            flow_catalog=flow_catalog,
            app_catalog=app_catalog,
        )

        return OperationIntent(
            session_id=str(session_id or "").strip(),
            snapshot_id=str(snapshot_id or "").strip(),
            supi=str(evidence.supi or "").strip(),
            app_id=selected_app_id,
            app_name=app_name,
            operation_type=str(decision.operation_type or "modify").strip() or "modify",
            urgency="Normal",
            raw_input=str(user_input or "").strip(),
            resolution_status=resolution_status,
            requested_domains=grounded_requested_domains,
            main_requested_domains=list(evidence.main_requested_domains or []),
            grounded_requested_domains=grounded_requested_domains,
            domain_revision_needed=domain_revision_needed,
            domain_revision_rationale=str(decision.domain_revision_rationale or "").strip(),
            domain_resolution=domain_resolution,
            domain_evidence=normalize_domain_evidence(
                directives.get("domain_evidence") or evidence.domain_evidence,
            ),
            control_semantics=control_semantics,
            mobility_intent=resolved_mobility_intent,
            grounding_evidence=self._build_grounding_evidence(
                evidence=evidence,
                directives=directives,
                selected_app_id=selected_app_id,
                flows=flows,
            ),
            flows=flows,
            qos_target_envelopes=qos_target_envelopes,
        )

    @staticmethod
    def _resolve_selected_app_name(*, app_catalog: List[Dict[str, Any]], selected_app_id: str) -> Optional[str]:
        selected_app = next(
            (item for item in app_catalog if str(item.get("app_id") or "").strip() == selected_app_id),
            None,
        )
        if selected_app is None:
            return None
        return str(selected_app.get("app_name") or "").strip() or None

    @staticmethod
    def _resolve_mobility_intent(
        *,
        decision: IntentAdvisorDecision,
    ) -> Dict[str, Any]:
        if isinstance(decision.mobility_intent, dict) and decision.mobility_intent:
            return dict(decision.mobility_intent)
        return {}

    def _resolve_selected_app_id(
        self,
        *,
        decision: IntentAdvisorDecision,
        selected_flow: Optional[Dict[str, Any]],
    ) -> str:
        selected_app_id = str(decision.selected_app_id or "").strip()
        if not selected_app_id and decision.flows:
            for flow in decision.flows:
                candidate_app_id = str(flow.app_id or "").strip()
                if candidate_app_id:
                    return candidate_app_id
        if not selected_app_id and selected_flow is not None:
            return str(selected_flow.get("app_id") or "").strip()
        return selected_app_id

    def _build_compiled_flows(
        self,
        *,
        evidence: IntentEvidence,
        decision: IntentAdvisorDecision,
        selected_app_id: str,
        flow_catalog: List[Dict[str, Any]],
        semantic_candidates: List[Dict[str, Any]],
    ) -> List[FlowSelector]:
        if list(evidence.requested_domains or []) == ["mobility"]:
            return []
        if decision.flows:
            return self._build_operation_flows_from_advisor_decision(
                advisor_flows=decision.flows,
                evidence=evidence,
                selected_app_id=selected_app_id,
                flow_catalog=flow_catalog,
                semantic_candidates=semantic_candidates,
            )
        return []

    @staticmethod
    def _resolve_operation_status(
        *,
        requested_domains: List[str],
        candidate_flows: List[FlowCandidateEvidence],
        flows: List[FlowSelector],
    ) -> str:
        if list(requested_domains or []) == ["mobility"]:
            return "resolved"
        if any(flow.resolution_status == "ambiguous" for flow in flows):
            return "ambiguous"
        if flows:
            return "resolved"
        return "unmatched" if not candidate_flows else "ambiguous"

    def _build_grounding_evidence(
        self,
        *,
        evidence: IntentEvidence,
        directives: Dict[str, Any],
        selected_app_id: str,
        flows: List[FlowSelector],
    ) -> GroundingEvidenceBundle:
        selected_flow_ids = {
            str(flow.flow_id or "").strip()
            for flow in flows
            if str(flow.flow_id or "").strip()
        }
        evidence_sources = {
            "domain_routing": list(normalize_domain_evidence(directives.get("domain_evidence") or evidence.domain_evidence).keys()),
            "grounding_tools": [
                item
                for item in [
                    "sm_grounding" if uses_sm_grounding(evidence.requested_domains) else "",
                    "am_grounding" if uses_am_grounding(evidence.requested_domains) else "",
                ]
                if item
            ],
        }
        evidence_sources = {
            key: [str(item).strip() for item in values if str(item).strip()]
            for key, values in evidence_sources.items()
            if values
        }
        return GroundingEvidenceBundle(
            grounded_supi=str(evidence.supi or "").strip(),
            grounded_apps=[
                {
                    "app_id": str(item.get("app_id") or "").strip(),
                    "app_name": str(item.get("app_name") or "").strip(),
                    "selected": str(item.get("app_id") or "").strip() == selected_app_id,
                }
                for item in (evidence.candidate_apps or [])
            ],
            grounded_flows=[
                {
                    "supi": candidate.supi,
                    "app_id": candidate.app_id,
                    "app_name": candidate.app_name,
                    "flow_id": str(candidate.flow_id or "").strip(),
                    "flow_name": candidate.flow_name,
                    "service_type": candidate.service_type,
                    "service_type_id": candidate.service_type_id,
                    "score": candidate.score,
                    "selected": str(candidate.flow_id or "").strip() in selected_flow_ids,
                }
                for candidate in evidence.candidate_flows
            ],
            grounded_mobility_targets={
                "summary": dict(evidence.am_context_summary or {}),
                "candidates": list(evidence.am_policy_candidates or []),
            },
            evidence_sources=evidence_sources,
        )

    def _resolve_selected_flow(
        self,
        *,
        decision: IntentAdvisorDecision,
        flow_catalog: List[Dict[str, Any]],
        semantic_candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        selected_flow_id = str(decision.selected_flow_id or "").strip()
        if selected_flow_id:
            return self._lookup_flow_record(
                flow_id=selected_flow_id,
                flow_catalog=flow_catalog,
                semantic_candidates=semantic_candidates,
            )
        return None

    def _build_operation_flows_from_advisor_decision(
        self,
        *,
        advisor_flows: List[FlowSelector],
        evidence: IntentEvidence,
        selected_app_id: str,
        flow_catalog: List[Dict[str, Any]],
        semantic_candidates: List[Dict[str, Any]],
    ) -> List[FlowSelector]:
        resolved: List[FlowSelector] = []
        for advisor_flow in advisor_flows:
            advisor_flow_id = str(advisor_flow.flow_id or "").strip()
            resolution_status = str(advisor_flow.resolution_status or "resolved").strip().lower() or "resolved"
            if resolution_status == "resolved" and advisor_flow_id and not flow_id_is_grounded(
                flow_id=advisor_flow_id,
                evidence=evidence,
            ):
                raise ValueError(f"resolved advisor flow_id is not grounded by evidence: {advisor_flow_id}")
            catalog_flow = self._lookup_flow_record(
                flow_id=advisor_flow_id,
                flow_catalog=flow_catalog,
                semantic_candidates=semantic_candidates,
            )
            resolved.append(
                FlowSelector.model_validate(
                    self._merge_advisor_flow_payload(
                        advisor_flow=advisor_flow,
                        catalog_flow=catalog_flow,
                        supi=evidence.supi,
                        selected_app_id=selected_app_id,
                    )
                )
            )
        return resolved

    @staticmethod
    def _lookup_flow_record(
        *,
        flow_id: str,
        flow_catalog: List[Dict[str, Any]],
        semantic_candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        normalized_flow_id = str(flow_id or "").strip()
        if not normalized_flow_id:
            return None
        catalog_match = next(
            (item for item in flow_catalog if str(item.get("flow_id") or "").strip() == normalized_flow_id),
            None,
        )
        if catalog_match is not None:
            return catalog_match
        return next(
            (item for item in semantic_candidates if str(item.get("flow_id") or "").strip() == normalized_flow_id),
            None,
        )

    def _merge_advisor_flow_payload(
        self,
        *,
        advisor_flow: FlowSelector,
        catalog_flow: Optional[Dict[str, Any]],
        supi: str,
        selected_app_id: str,
    ) -> Dict[str, Any]:
        merged_payload: Dict[str, Any] = {}
        if catalog_flow is not None:
            merged_payload = self._build_flow_selector_from_catalog(catalog_flow).model_dump(mode="json")
        semantic_keys = {
            "supi",
            "app_id",
            "app_name",
            "flow_id",
            "target_type",
            "name",
            "service_type",
            "service_type_id",
            "description",
            "resolution_status",
            "resolution_candidates",
        }
        for key, value in advisor_flow.model_dump(mode="json").items():
            if key not in semantic_keys:
                continue
            if isinstance(value, str):
                if value.strip():
                    merged_payload[key] = value
            elif isinstance(value, list):
                if value:
                    merged_payload[key] = list(value)
            elif value is not None:
                merged_payload[key] = value
        merged_payload["supi"] = str(merged_payload.get("supi") or supi or "").strip()
        merged_payload["app_id"] = str(merged_payload.get("app_id") or selected_app_id or "").strip()
        if not str(merged_payload.get("name") or "").strip():
            merged_payload["name"] = str(merged_payload.get("flow_id") or "").strip()
        if not str(merged_payload.get("resolution_status") or "").strip():
            merged_payload["resolution_status"] = "resolved" if str(merged_payload.get("flow_id") or "").strip() else "unmatched"
        return merged_payload

    @staticmethod
    def _build_flow_selector_from_catalog(flow: Dict[str, Any]) -> FlowSelector:
        service = flow.get("service") if isinstance(flow.get("service"), dict) else {}
        sla = flow.get("sla") if isinstance(flow.get("sla"), dict) else {}
        allocation = flow.get("allocation") if isinstance(flow.get("allocation"), dict) else {}
        traffic = flow.get("traffic") if isinstance(flow.get("traffic"), dict) else {}
        five_tuple = traffic.get("five_tuple")
        return FlowSelector(
            supi=str(flow.get("supi") or "").strip(),
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
            five_tuple=list(five_tuple) if isinstance(five_tuple, (list, tuple)) else None,
            current_bw_ul=allocation.get("allocated_bandwidth_ul"),
            current_bw_dl=allocation.get("allocated_bandwidth_dl"),
            resolution_status="resolved",
        )
