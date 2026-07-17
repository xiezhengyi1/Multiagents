from __future__ import annotations

import re
from typing import Any, List

from ...domain.intent_encoding import AM_GROUNDING_TOOLS, SM_GROUNDING_TOOLS, VALID_DOMAINS
from .common import flow_id_is_grounded, mobility_request_mentions_specific_targets
from .contracts import IntentEvidence
from ...domain.policy_plan import GroundingDecision


class IntentGroundingValidator:
    _SUBSCRIPTION_CHANGE_PATTERN = re.compile(
        r"(?:add|enable|provision|subscribe|subscription|开通|订阅|增加.*(?:切片|nssai)|变更.*(?:订阅|签约))",
        re.IGNORECASE,
    )

    @classmethod
    def _user_requests_subscription_change(cls, user_input: str) -> bool:
        return bool(cls._SUBSCRIPTION_CHANGE_PATTERN.search(str(user_input or "")))

    def validate_intent_grounding(
        self,
        *,
        evidence: IntentEvidence,
        grounding_tools: List[str],
        grounding_decision: GroundingDecision | None = None,
    ) -> List[str]:
        errors: List[str] = []
        requested_domains = {
            str(item or "").strip().lower()
            for item in (evidence.requested_domains or [])
            if str(item or "").strip()
        }
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
        if requested_domains == {"mobility"}:
            decision_has_am_context = self._grounding_decision_has_grounded_am_policy(grounding_decision)
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
    def _grounding_decision_has_grounded_am_policy(grounding_decision: GroundingDecision | None) -> bool:
        if grounding_decision is None:
            return False
        mobility_intent: Any = grounding_decision.mobility_intent or {}
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

    def validate_grounding_decision(
        self,
        *,
        evidence: IntentEvidence,
        grounding_decision: GroundingDecision,
    ) -> List[str]:
        errors: List[str] = []
        requested_domains = {
            str(item or "").strip().lower()
            for item in (evidence.requested_domains or [])
            if str(item or "").strip()
        }
        if requested_domains and not requested_domains.issubset(VALID_DOMAINS):
            errors.append(f"Main requested_domains contains unsupported values: {sorted(requested_domains)}")
        requires_slice_change = any(
            bool(constraint.require_slice_change)
            for constraint in (grounding_decision.qos_operation_constraints or [])
        )
        if requires_slice_change:
            authorization = grounding_decision.slice_migration_authorization
            valid_decisions = {
                "migration_pending_target_authorization",
                "migration_authorized",
                "blocked_by_subscription_entitlement",
                "blocked_requires_subscription_provisioning",
                "evidence_missing",
            }
            if not evidence.subscription_summary:
                errors.append("slice migration requires get_ue_slice_subscription entitlement evidence before final intent")
            if str(authorization.decision or "").strip() not in valid_decisions:
                errors.append("slice migration requires an explicit slice_migration_authorization decision")
            if not str(authorization.authority or "").strip():
                errors.append("slice migration authorization must name its evidence authority")
            if not authorization.authorized_snssais and not authorization.subscription_change_required:
                errors.append("slice migration authorization must preserve authorized S-NSSAI evidence or request provisioning")
            explicit_subscription_change = self._user_requests_subscription_change(evidence.user_input)
            decision = str(authorization.decision or "").strip()
            if authorization.subscription_change_required and not explicit_subscription_change:
                errors.append(
                    "slice migration cannot request subscription provisioning unless the user explicitly asks to add or change the subscription"
                )
            if decision == "blocked_by_subscription_entitlement" and authorization.subscription_change_required:
                errors.append(
                    "blocked_by_subscription_entitlement must not request subscription provisioning"
                )
            if decision in {"migration_pending_target_authorization", "evidence_missing"} and not explicit_subscription_change:
                errors.append(
                    "unverified slice migration is blocked: the user did not request subscription provisioning; keep the serving slice or return an explicit blocking question"
                )
            if decision == "migration_authorized":
                authorized = {str(item or "").strip() for item in authorization.authorized_snssais if str(item or "").strip()}
                targets = {str(item or "").strip() for item in authorization.target_snssais if str(item or "").strip()}
                if not authorized:
                    errors.append("migration_authorized requires non-empty subscription entitlement evidence")
                elif targets and not targets.issubset(authorized):
                    errors.append("migration_authorized target S-NSSAI is absent from subscription entitlement evidence")
        if requested_domains == {"mobility"} and grounding_decision.flows:
            errors.append("Mobility-only GroundingDecision must not include QoS flows.")
        if "qos" not in requested_domains:
            return errors

        if not grounding_decision.flows:
            errors.append("QoS GroundingDecision must include grounded target flows.")
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
                flow for flow in grounding_decision.flows
                if (
                    str(flow.name or "").strip() == flow_name
                    or grounded_flow_name_by_id.get(str(flow.flow_id or "").strip()) == flow_name
                )
            ]
            if not matching_flows:
                errors.append(
                    f"explicitly named QoS flow '{flow_name}' must appear in advisor decision flows as resolved or unresolved"
                )

        for index, flow in enumerate(grounding_decision.flows):
            resolution_status = str(flow.resolution_status or "resolved").strip().lower() or "resolved"
            flow_id = str(flow.flow_id or "").strip()
            if resolution_status == "resolved" and not flow_id:
                errors.append(f"QoS GroundingDecision flow[{index}] is resolved but missing flow_id.")
            if resolution_status == "resolved" and flow_id and not flow_id_is_grounded(
                flow_id=flow_id,
                evidence=evidence,
            ):
                errors.append(
                    f"QoS GroundingDecision flow[{index}] resolved flow_id={flow_id} is not grounded by catalog/search evidence."
                )
            if resolution_status == "unresolved" and not str(flow.name or "").strip():
                errors.append(
                    f"QoS GroundingDecision unresolved flow[{index}] must preserve the named target in name."
                )
        resolved_flow_ids = {
            str(flow.flow_id or "").strip()
            for flow in grounding_decision.flows
            if str(flow.resolution_status or "resolved").strip().lower() == "resolved"
            and str(flow.flow_id or "").strip()
        }
        flow_by_id = {
            str(flow.flow_id or "").strip(): flow
            for flow in grounding_decision.flows
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
        return errors


def _missing_baseline_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())
