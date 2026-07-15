from __future__ import annotations

from typing import Any, List

from ...domain.intent_encoding import AM_GROUNDING_TOOLS, SM_GROUNDING_TOOLS, VALID_DOMAINS
from .common import flow_id_is_grounded, mobility_request_mentions_specific_targets
from .contracts import IntentEvidence
from ...domain.policy_plan import OperationIntent


class IntentGroundingValidator:
    def validate_intent_grounding(
        self,
        *,
        evidence: IntentEvidence,
        grounding_tools: List[str],
        operation_intent: OperationIntent | None = None,
    ) -> List[str]:
        errors: List[str] = []
        requested_domains = {str(item or "").strip().lower() for item in (evidence.requested_domains or []) if str(item or "").strip()}
        used_grounding_tools = {str(item or "").strip() for item in (grounding_tools or []) if str(item or "").strip()}
        if requested_domains == {"mobility"} and (used_grounding_tools & SM_GROUNDING_TOOLS):
            errors.append("mobility-only intent must not call SM grounding tools")
        if requested_domains == {"qos"} and (used_grounding_tools & AM_GROUNDING_TOOLS):
            errors.append("QoS-only intent must not call AM grounding tools")
        if "qos" in requested_domains and str(evidence.supi or "").strip():
            if not evidence.catalog_evidence_observed:
                errors.append(
                    "QoS intent with a known SUPI requires get_sm_ue_flow_catalog catalog evidence before final intent"
                )
        if list(evidence.requested_domains or []) == ["mobility"]:
            decision_has_am_context = self._operation_intent_has_grounded_am_policy(operation_intent)
            if not evidence.am_context_summary and not decision_has_am_context:
                errors.append("mobility-only intent requires grounded AM policy context before returning final intent")
            if mobility_request_mentions_specific_targets(evidence.user_input):
                if not evidence.am_policy_candidates and not decision_has_am_context:
                    errors.append(
                        "mobility intent that names association/RFSP/NSSAI/service-area/access targets requires search_am_policy_targets evidence"
                    )
            return errors
        named_flow_request = bool(
            str(evidence.explicit_app_id or "").strip()
            or str(evidence.explicit_app_name or "").strip()
            or str(evidence.explicit_flow_id or "").strip()
            or str(evidence.explicit_flow_name or "").strip()
            or list(evidence.explicit_flow_targets or [])
        )
        if (
            "qos" in requested_domains
            and named_flow_request
            and not evidence.candidate_flows
            and (evidence.catalog_payload or {}).get("flow_catalog")
            and "search_sm_flow_targets" not in used_grounding_tools
        ):
            errors.append("named QoS flow request requires search_sm_flow_targets before returning final intent")
        if "qos" in requested_domains and not evidence.candidate_flows and not grounding_tools:
            errors.append("unresolved QoS intent requires at least one grounding tool call")
        return errors

    @staticmethod
    def _operation_intent_has_grounded_am_policy(operation_intent: OperationIntent | None) -> bool:
        if operation_intent is None:
            return False
        mobility_intent: Any = operation_intent.mobility_intent or {}
        if not isinstance(mobility_intent, dict):
            return False
        association_id = str(
            mobility_intent.get("association_id")
            or mobility_intent.get("current_association_id")
            or ""
        ).strip()
        rfsp = mobility_intent.get("current_rfsp", mobility_intent.get("rfsp"))
        allowed_snssais = mobility_intent.get("current_allowed_snssais", mobility_intent.get("allowed_snssais"))
        return bool(association_id and rfsp is not None and allowed_snssais)

    def validate_operation_intent(
        self,
        *,
        evidence: IntentEvidence,
        operation_intent: OperationIntent,
    ) -> List[str]:
        errors: List[str] = []
        requested_domains = {str(item or "").strip().lower() for item in (evidence.requested_domains or []) if str(item or "").strip()}
        grounded_domains = {
            str(item or "").strip().lower()
            for item in (operation_intent.requested_domains or evidence.requested_domains or [])
            if str(item or "").strip()
        }
        if operation_intent.domain_resolution not in {"confirmed", "narrowed", "widened", "cannot_confirm"}:
            errors.append("domain_resolution must be confirmed, narrowed, widened, or cannot_confirm")
        if operation_intent.domain_resolution == "cannot_confirm" and not operation_intent.open_questions:
            errors.append("cannot_confirm domain resolution requires open_questions")
        if grounded_domains and not grounded_domains.issubset(VALID_DOMAINS):
            errors.append(f"requested_domains contains unsupported values: {sorted(grounded_domains)}")
        if requested_domains == {"mobility"} and operation_intent.flows:
            errors.append("Mobility-only OperationIntent must not include QoS flows.")
        if "qos" not in requested_domains:
            return errors

        if not operation_intent.flows:
            errors.append("QoS OperationIntent must include grounded target flows.")
            return errors

        explicit_target_names = {
            str(item.flow_name or "").strip()
            for item in (evidence.explicit_flow_targets or [])
            if str(item.flow_name or "").strip()
        }
        grounded_flow_name_by_id = {
            str(item.flow_id or "").strip(): str(item.flow_name or "").strip()
            for item in (evidence.candidate_flows or [])
            if str(item.flow_id or "").strip() and str(item.flow_name or "").strip()
        }
        catalog_payload = evidence.catalog_payload or {}
        for item in catalog_payload.get("flow_catalog") or []:
            if not isinstance(item, dict):
                continue
            flow_id = str(item.get("flow_id") or "").strip()
            flow_name = str(item.get("flow_name") or "").strip()
            if flow_id and flow_name:
                grounded_flow_name_by_id.setdefault(flow_id, flow_name)
        for flow_name in explicit_target_names:
            matching_flows = [
                flow for flow in operation_intent.flows
                if (
                    str(flow.name or "").strip() == flow_name
                    or grounded_flow_name_by_id.get(str(flow.flow_id or "").strip()) == flow_name
                )
            ]
            if not matching_flows:
                errors.append(
                    f"explicitly named QoS flow '{flow_name}' must appear in advisor decision flows as resolved or unresolved"
                )

        for index, flow in enumerate(operation_intent.flows):
            resolution_status = str(flow.resolution_status or "resolved").strip().lower() or "resolved"
            flow_id = str(flow.flow_id or "").strip()
            if resolution_status == "resolved" and not flow_id:
                errors.append(f"QoS OperationIntent flow[{index}] is resolved but missing flow_id.")
            if resolution_status == "resolved" and flow_id and not flow_id_is_grounded(
                flow_id=flow_id,
                evidence=evidence,
            ):
                errors.append(
                    f"QoS OperationIntent flow[{index}] resolved flow_id={flow_id} is not grounded by catalog/search evidence."
                )
            if resolution_status == "unresolved" and not str(flow.name or "").strip():
                errors.append(
                    f"QoS OperationIntent unresolved flow[{index}] must preserve the named target in name."
                )
        resolved_flow_ids = {
            str(flow.flow_id or "").strip()
            for flow in operation_intent.flows
            if str(flow.resolution_status or "resolved").strip().lower() == "resolved"
            and str(flow.flow_id or "").strip()
        }
        envelope_by_flow_id = {
            str(envelope.flow_id or "").strip(): envelope
            for envelope in (operation_intent.qos_target_envelopes or [])
            if str(envelope.flow_id or "").strip()
        }
        flow_by_id = {
            str(flow.flow_id or "").strip(): flow
            for flow in operation_intent.flows
            if str(flow.flow_id or "").strip()
        }
        flow_baseline_fields = (
            "service_type_id",
            "bw_ul",
            "bw_dl",
            "gbr_ul",
            "gbr_dl",
            "lat",
            "loss_req",
            "jitter_req",
            "priority",
            "current_slice_snssai",
        )
        envelope_baseline_fields = (
            "baseline_priority",
            "baseline_latency_ms",
            "baseline_jitter_ms",
            "baseline_packet_error_rate",
            "baseline_max_br_ul_mbps",
            "baseline_max_br_dl_mbps",
            "baseline_gbr_ul_mbps",
            "baseline_gbr_dl_mbps",
        )
        for flow_id in sorted(resolved_flow_ids):
            flow = flow_by_id.get(flow_id)
            if flow is not None:
                missing_flow_fields = [
                    field
                    for field in flow_baseline_fields
                    if _missing_baseline_value(getattr(flow, field, None))
                ]
                if missing_flow_fields:
                    errors.append(
                        f"incomplete QoS flow selector for resolved flow_id={flow_id}: "
                        + ", ".join(missing_flow_fields)
                    )
            envelope = envelope_by_flow_id.get(flow_id)
            if envelope is None:
                errors.append(f"missing QoS baseline envelope for resolved flow_id={flow_id}")
                continue
            missing_envelope_fields = [
                field
                for field in envelope_baseline_fields
                if _missing_baseline_value(getattr(envelope, field, None))
            ]
            if len(missing_envelope_fields) == len(envelope_baseline_fields):
                errors.append(f"missing QoS baseline values for resolved flow_id={flow_id}")
            elif missing_envelope_fields:
                errors.append(
                    f"incomplete QoS baseline for resolved flow_id={flow_id}: "
                    + ", ".join(missing_envelope_fields)
                )
        stages = list(operation_intent.control_semantics.stages or [])
        if not stages:
            errors.append("QoS OperationIntent must include IEA-owned control_semantics stages.")
            return errors
        active_flow_ids = {
            str(flow_id or "").strip()
            for stage in stages
            for flow_id in (stage.active_flow_ids or [])
            if str(flow_id or "").strip()
        }
        if resolved_flow_ids and not active_flow_ids:
            errors.append("QoS OperationIntent control_semantics must include active_flow_ids for grounded stage targets.")
        unknown_active_ids = active_flow_ids - resolved_flow_ids
        if unknown_active_ids:
            errors.append(
                "QoS OperationIntent control_semantics active_flow_ids must reference flows: "
                + ", ".join(sorted(unknown_active_ids))
            )
        return errors


def _missing_baseline_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())
