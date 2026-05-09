from __future__ import annotations

from typing import List

from .common import AM_GROUNDING_TOOLS, SM_GROUNDING_TOOLS, VALID_DOMAINS, flow_id_is_grounded, mobility_request_mentions_specific_targets
from .contracts import IntentAdvisorDecision, IntentEvidence


class IntentGroundingValidator:
    def validate_intent_grounding(
        self,
        *,
        evidence: IntentEvidence,
        grounding_tools: List[str],
    ) -> List[str]:
        errors: List[str] = []
        requested_domains = {str(item or "").strip().lower() for item in (evidence.requested_domains or []) if str(item or "").strip()}
        used_grounding_tools = {str(item or "").strip() for item in (grounding_tools or []) if str(item or "").strip()}
        if requested_domains == {"mobility"} and (used_grounding_tools & SM_GROUNDING_TOOLS):
            errors.append("mobility-only intent must not call SM grounding tools")
        if requested_domains == {"qos"} and (used_grounding_tools & AM_GROUNDING_TOOLS):
            errors.append("QoS-only intent must not call AM grounding tools")
        if list(evidence.requested_domains or []) == ["mobility"]:
            if not evidence.am_context_summary:
                errors.append("mobility-only intent requires grounded AM policy context before returning final intent")
            if mobility_request_mentions_specific_targets(evidence.user_input):
                if not evidence.am_policy_candidates:
                    errors.append(
                        "mobility intent that names association/RFSP/NSSAI/service-area/access targets requires search_am_policy_targets evidence"
                    )
            return errors
        named_flow_request = (
            "/" in str(evidence.user_input or "")
            and not str(evidence.explicit_app_id or "").strip()
            and not str(evidence.explicit_flow_id or "").strip()
            and not list(evidence.explicit_flow_targets or [])
        )
        if (
            "qos" in requested_domains
            and named_flow_request
            and not evidence.candidate_flows
            and "search_sm_flow_targets" not in used_grounding_tools
        ):
            errors.append("named QoS flow request requires search_sm_flow_targets before returning final intent")
        if "qos" in requested_domains and not evidence.candidate_flows and not grounding_tools:
            errors.append("unresolved QoS intent requires at least one grounding tool call")
        return errors

    def validate_advisor_decision(
        self,
        *,
        evidence: IntentEvidence,
        decision: IntentAdvisorDecision,
    ) -> List[str]:
        errors: List[str] = []
        requested_domains = {str(item or "").strip().lower() for item in (evidence.requested_domains or []) if str(item or "").strip()}
        grounded_domains = {
            str(item or "").strip().lower()
            for item in (decision.grounded_requested_domains or evidence.requested_domains or [])
            if str(item or "").strip()
        }
        if decision.domain_resolution not in {"confirmed", "narrowed", "widened", "cannot_confirm"}:
            errors.append("domain_resolution must be confirmed, narrowed, widened, or cannot_confirm")
        if decision.domain_resolution == "cannot_confirm" and not str(decision.domain_revision_rationale or "").strip():
            errors.append("cannot_confirm domain resolution requires domain_revision_rationale")
        if grounded_domains and not grounded_domains.issubset(VALID_DOMAINS):
            errors.append(f"grounded_requested_domains contains unsupported values: {sorted(grounded_domains)}")
        if requested_domains == {"mobility"} and decision.flows:
            errors.append("Mobility-only advisor decision must not include QoS flows.")
        if "qos" not in requested_domains:
            return errors

        if not decision.flows:
            errors.append("QoS advisor decision must include grounded target flows.")
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
                flow for flow in decision.flows
                if (
                    str(flow.name or "").strip() == flow_name
                    or grounded_flow_name_by_id.get(str(flow.flow_id or "").strip()) == flow_name
                )
            ]
            if not matching_flows:
                errors.append(
                    f"explicitly named QoS flow '{flow_name}' must appear in advisor decision flows as resolved or unresolved"
                )

        for index, flow in enumerate(decision.flows):
            resolution_status = str(flow.resolution_status or "resolved").strip().lower() or "resolved"
            flow_id = str(flow.flow_id or "").strip()
            if resolution_status == "resolved" and not flow_id:
                errors.append(f"QoS advisor flow[{index}] is resolved but missing flow_id.")
            if resolution_status == "resolved" and flow_id and not flow_id_is_grounded(
                flow_id=flow_id,
                evidence=evidence,
            ):
                errors.append(
                    f"QoS advisor flow[{index}] resolved flow_id={flow_id} is not grounded by catalog/search evidence."
                )
        return errors
