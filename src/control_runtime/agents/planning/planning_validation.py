from __future__ import annotations

from datetime import date, datetime
from enum import Enum
import json
import re
from typing import Any, Dict, List

from pydantic import BaseModel

from ...domain.collaboration import PlanningRequest
from ...domain.control_plane import ControlDomain, DomainStatus
from ...domain.policy_plan import PolicyDraft, PolicyPlanDraft
from ...context.evidence import build_slice_snssai
from .response_models import OsaAdvisorOutput
from model.PcfAmPolicyControl import PcfAmPolicyControlPolicyAssociation
from model.SmPolicyDecision import SmPolicyDecision
from model.UrspRuleRequest import UrspRuleRequest


def json_friendly(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return json_friendly(value.model_dump(mode="json", by_alias=False))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_friendly(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_friendly(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def normalize_app_id(app_id: Any) -> str:
    text = str(app_id or "").strip()
    if not text:
        return ""
    if text.startswith(("app_", "app-")):
        return f"app-{text[4:].replace('_', '-')}"
    if re.fullmatch(r"app\d+", text, flags=re.IGNORECASE):
        return f"app-{text[3:]}"
    return text.replace("_", "-")


def _require_policy_id(policy_id: Any, *, policy_type: str) -> str:
    normalized = str(policy_id or "").strip()
    if not normalized:
        raise ValueError(f"{policy_type} is missing policy_id")
    return normalized


def _require_supi(supi: Any) -> str:
    normalized = str(supi or "").strip()
    if not normalized:
        raise ValueError("PolicyPlanDraft is missing authoritative supi")
    return normalized


def _normalize_sm_policy_details(details: Dict[str, Any], *, flow_id: str, app_id: str) -> Dict[str, Any]:
    data = json_friendly(details)
    if not isinstance(data, dict):
        raise ValueError("SmPolicyDecision is missing policy_details")
    if not flow_id:
        raise ValueError("SmPolicyDecision is missing flow_id")
    if not app_id:
        raise ValueError("SmPolicyDecision is missing app_id")
    return json_friendly(SmPolicyDecision.model_validate(data))


def _normalize_ursp_policy_details(details: Dict[str, Any], *, target_type: str, flow_id: str | None) -> Dict[str, Any]:
    data = json_friendly(details)
    if not isinstance(data, dict):
        raise ValueError("UrspRuleRequest is missing policy_details")
    if target_type == "flow" and not str(flow_id or "").strip():
        raise ValueError("flow-scoped URSP policy is missing flow_id")
    return json_friendly(UrspRuleRequest.model_validate(data))


def _normalize_am_policy_details(details: Dict[str, Any], *, supi: str) -> Dict[str, Any]:
    data = json_friendly(details)
    if not isinstance(data, dict):
        raise ValueError("PcfAmPolicyControlPolicyAssociation is missing policy_details")
    request = data.get("request")
    if not isinstance(request, dict):
        raise ValueError("PcfAmPolicyControlPolicyAssociation requires a request object")
    request_supi = str(request.get("supi") or "").strip()
    if request_supi != supi:
        raise ValueError("AM policy request.supi must match authoritative supi")
    return json_friendly(PcfAmPolicyControlPolicyAssociation.model_validate(data))


def normalize_policy_plan_draft(draft: PolicyPlanDraft) -> PolicyPlanDraft:
    base_supi = _require_supi(draft.supi)
    normalized_policies: List[PolicyDraft] = []
    for index, policy in enumerate(draft.all_policies, start=1):
        policy_type = str(policy.policy_type or "").strip()
        if not policy_type:
            raise ValueError(f"Policy #{index} is missing policy_type")
        supi = _require_supi(policy.supi or base_supi)
        app_id = normalize_app_id(policy.app_id)
        flow_id = str(policy.flow_id or "").strip() or None
        target_type = str(policy.target_type or "").strip().lower()
        if not target_type:
            raise ValueError(f"Policy #{index} is missing target_type")

        if policy_type == "SmPolicyDecision":
            if target_type != "flow":
                raise ValueError("SmPolicyDecision target_type must be flow")
            norm_details = _normalize_sm_policy_details(policy.policy_details, flow_id=flow_id or "", app_id=app_id)
        elif policy_type == "UrspRuleRequest":
            if target_type not in {"flow", "app"}:
                raise ValueError("UrspRuleRequest target_type must be flow or app")
            norm_details = _normalize_ursp_policy_details(policy.policy_details, target_type=target_type, flow_id=flow_id)
        elif policy_type == "PcfAmPolicyControlPolicyAssociation":
            if target_type != "ue":
                raise ValueError("PcfAmPolicyControlPolicyAssociation target_type must be ue")
            if flow_id is not None:
                raise ValueError("PcfAmPolicyControlPolicyAssociation must not include flow_id")
            if app_id:
                raise ValueError("PcfAmPolicyControlPolicyAssociation must not include app_id")
            norm_details = _normalize_am_policy_details(policy.policy_details, supi=supi)
        else:
            raise ValueError(f"Unsupported policy_type: {policy_type}")

        normalized_policies.append(
            PolicyDraft(
                recommended_actions=[],
                supi=supi,
                app_id=app_id,
                flow_id=flow_id,
                target_type=target_type,
                policy_id=_require_policy_id(policy.policy_id, policy_type=policy_type),
                policy_type=policy_type,
                resource_keys=[str(item or "").strip() for item in (policy.resource_keys or []) if str(item or "").strip()],
                policy_details=norm_details,
            )
        )

    normalized_partial_policies: List[PolicyDraft] = []
    for index, policy in enumerate(draft.partial_policies, start=1):
        policy_type = str(policy.policy_type or "").strip()
        if not policy_type:
            raise ValueError(f"partial_policies[{index}] is missing policy_type")
        normalized_partial_policies.append(
            PolicyDraft(
                recommended_actions=[],
                supi=_require_supi(policy.supi or base_supi),
                app_id=normalize_app_id(policy.app_id),
                flow_id=str(policy.flow_id or "").strip() or None,
                target_type=str(policy.target_type or "").strip().lower() or "flow",
                policy_id=_require_policy_id(policy.policy_id, policy_type=policy_type),
                policy_type=policy_type,
                resource_keys=[str(item or "").strip() for item in (policy.resource_keys or []) if str(item or "").strip()],
                policy_details=json_friendly(policy.policy_details),
            )
        )

    return PolicyPlanDraft(
        supi=base_supi,
        session_id=str(draft.session_id or "").strip(),
        snapshot_id=str(draft.snapshot_id or "").strip(),
        planning_status=str(draft.planning_status or "executable_plan").strip(),
        optimizer_result=json_friendly(draft.optimizer_result),
        all_policies=normalized_policies,
        partial_policies=normalized_partial_policies,
        missing_evidence=[str(item) for item in (draft.missing_evidence or []) if str(item or "").strip()],
        blocked_targets=[str(item) for item in (draft.blocked_targets or []) if str(item or "").strip()],
        upstream_requests=[str(item) for item in (draft.upstream_requests or []) if str(item or "").strip()],
        planner_conflicts=[str(item) for item in (draft.planner_conflicts or []) if str(item or "").strip()],
        open_questions=[item.model_copy(deep=True) for item in draft.open_questions],
        planning_rationale=draft.planning_rationale.model_copy(deep=True),
    )



_json_friendly = json_friendly
_normalize_app_id = normalize_app_id


def _main_retry_scope(planning_request: PlanningRequest) -> str:
    retry_scope = planning_request.context.shared_context.main_intent.retry_scope
    return str(getattr(retry_scope, "value", retry_scope) or "").strip().lower()


class PlanningAdvisorValidator:
    def validate_advisor_output(
        self,
        *,
        advisor_output: OsaAdvisorOutput,
        planning_request: PlanningRequest,
        grounding_tools: List[str],
        planning_tool_evidence: Dict[str, Any] | None = None,
    ) -> List[str]:
        errors: List[str] = []
        domains = {
            str(item.value if hasattr(item, "value") else item or "").strip().lower()
            for item in (planning_request.context.shared_context.main_intent.requested_domains or [])
            if str(item.value if hasattr(item, "value") else item or "").strip()
        }
        normalized_tool_evidence = dict(planning_tool_evidence or {})
        retry_scope = _main_retry_scope(planning_request)
        preserved_app_id = self._preserved_app_id(planning_request)
        preserved_flow_ids = self._preserved_flow_ids(planning_request)
        preserved_flow_app_ids = self._preserved_flow_app_ids(planning_request)
        planning_status = str(advisor_output.planning_status or "").strip().lower()

        has_sm = bool(advisor_output.sm_policies)
        has_am = advisor_output.am_policy is not None
        has_ursp = bool(advisor_output.ursp_policies)
        migration_requirements = self._required_slice_change_constraints(planning_request)
        migration_blocked = self._migration_is_blocked_by_iea(planning_request)
        optimizer_preview = self._latest_optimizer_preview(normalized_tool_evidence)
        migration_target_unauthorized = self._migration_target_is_unauthorized(
            planning_request=planning_request,
            optimizer_preview=optimizer_preview,
        )

        if migration_blocked:
            if planning_status != "partial_plan":
                errors.append(
                    "IEA blocked target slice migration; return partial_plan for entitlement-limited best-effort delivery"
                )
            if has_am or has_ursp:
                errors.append(
                    "blocked slice migration may deliver QoS-only SM policies but must not include AM or URSP policies"
                )
            if has_sm:
                errors.extend(
                    self._validate_entitlement_limited_qos_delivery(
                        optimizer_preview,
                        planning_request=planning_request,
                    )
                )
        if migration_target_unauthorized and not migration_blocked:
            if planning_status == "executable_plan":
                errors.append("optimizer-selected slice is absent from subscription entitlement evidence")
            if has_sm or has_am or has_ursp:
                errors.append("unauthorized slice migration must not include executable AM, SM, or URSP policies")
            return errors

        if has_sm and "qos" not in domains:
            errors.append("advisor emitted sm_policies outside qos-active planning")
        if has_am and "mobility" not in domains:
            errors.append("advisor emitted am_policy outside mobility-active planning")
        if planning_status == "executable_plan" and "qos" in domains and not has_sm:
            errors.append("qos-active planning requires sm_policies")
        if planning_status == "executable_plan" and "mobility" in domains and not has_am:
            errors.append("mobility-active planning requires am_policy")
        if migration_requirements and planning_status == "executable_plan" and not migration_blocked:
            if not has_sm:
                errors.append("authorized slice migration requires an SM policy")
            if not has_am:
                errors.append("authorized slice migration requires an AM policy")
            if has_am:
                errors.extend(
                    self._validate_am_targets_for_slice_migration(
                        planning_request=planning_request,
                        optimizer_preview=self._latest_optimizer_preview(normalized_tool_evidence),
                        am_policy=advisor_output.am_policy,
                    )
                )
        if has_sm:
            errors.extend(
                self._validate_sm_policy_target_bindings(
                    advisor_output.sm_policies,
                    planning_request=planning_request,
                )
            )
        mobility_context = self._latest_mobility_context(normalized_tool_evidence)
        if (
            "qos" in domains
            and planning_status != "executable_plan"
            and self._optimizer_preview_has_grounded_qos_assignments(
                optimizer_preview,
                flow_ids=preserved_flow_ids,
            )
            and not migration_target_unauthorized
            and not migration_blocked
            and not self._validate_required_slice_changes(
                optimizer_preview,
                planning_request=planning_request,
            )
        ):
            errors.append(
                "approved optimizer preview with grounded QoS assignments must return executable_plan "
                "with sm_policies; do not request upstream SNSSAI or RAG validation"
            )
        if has_sm and not optimizer_preview:
            errors.append("sm_policies require a parseable optimizer preview payload")
        elif has_sm:
            preview_errors = self._validate_optimizer_preview_payload(optimizer_preview)
            errors.extend(preview_errors)
            preview_infeasible = any(
                any(token in str(error_text or "").strip().lower() for token in ("infeasible", "incomplete_context"))
                for error_text in preview_errors
            )
            if not preview_infeasible:
                for spec in advisor_output.sm_policies:
                    if not self._extract_qos_resource_keys(optimizer_preview, flow_id=spec.flow_id):
                        errors.append(
                            f"optimizer preview does not contain a grounded QoS assignment for flow_id={spec.flow_id}"
                        )
                if not migration_blocked:
                    errors.extend(
                        self._validate_required_slice_changes(
                            optimizer_preview,
                            planning_request=planning_request,
                        )
                    )
        if has_am and not mobility_context:
            errors.append("am_policy requires a parseable mobility context payload")
        if retry_scope == "target_stable":
            if preserved_flow_app_ids:
                for index, spec in enumerate(advisor_output.sm_policies):
                    flow_id = str(spec.flow_id or "").strip()
                    preserved_spec_app_id = preserved_flow_app_ids.get(flow_id)
                    if preserved_spec_app_id and _normalize_app_id(spec.app_id) != preserved_spec_app_id:
                        errors.append(
                            "target-stable retry "
                            f"sm_policies[{index}] must preserve app_id={preserved_spec_app_id} "
                            f"for flow_id={flow_id}; got {spec.app_id}"
                        )
            elif preserved_app_id:
                for index, spec in enumerate(advisor_output.sm_policies):
                    if _normalize_app_id(spec.app_id) != preserved_app_id:
                        errors.append(
                            f"target-stable retry sm_policies[{index}] must preserve app_id={preserved_app_id}; got {spec.app_id}"
                        )
            if preserved_flow_ids:
                advisor_flow_ids = {str(item.flow_id or "").strip() for item in advisor_output.sm_policies if str(item.flow_id or "").strip()}
                if advisor_flow_ids and advisor_flow_ids != preserved_flow_ids:
                    errors.append(
                        "target-stable retry sm_policies must preserve grounded flow_ids "
                        f"{sorted(preserved_flow_ids)}; got {sorted(advisor_flow_ids)}"
                    )
        return errors

    @staticmethod
    def _normalize_snssai_key(value: Any) -> str:
        if isinstance(value, dict):
            try:
                sst = int(value.get("sst"))
            except (TypeError, ValueError):
                return ""
            sd = str(value.get("sd") or "").strip().lower()
            if not 0 <= sst <= 255 or len(sd) != 6 or any(char not in "0123456789abcdef" for char in sd):
                return ""
            return f"{sst:02x}{sd}"
        text = str(value or "").strip().lower()
        if len(text) == 8 and all(char in "0123456789abcdef" for char in text):
            return text
        return ""

    @classmethod
    def _migration_is_blocked_by_iea(cls, planning_request: PlanningRequest) -> bool:
        authorization = planning_request.grounding_decision.slice_migration_authorization
        return str(authorization.decision or "").strip() in {
            "blocked_by_subscription_entitlement",
            "blocked_requires_subscription_provisioning",
            "evidence_missing",
        }

    @classmethod
    def _migration_target_is_unauthorized(
        cls,
        *,
        planning_request: PlanningRequest,
        optimizer_preview: Dict[str, Any],
    ) -> bool:
        requirements = cls._required_slice_change_constraints(planning_request)
        if not requirements:
            return False
        authorized = {
            key
            for key in (
                cls._normalize_snssai_key(item)
                for item in planning_request.grounding_decision.slice_migration_authorization.authorized_snssais
            )
            if key
        }
        if not authorized:
            return True
        for requirement in requirements:
            selected = cls._normalize_snssai_key(cls._extract_selected_slice(optimizer_preview, flow_id=requirement["flow_id"]))
            if selected and selected != cls._normalize_snssai_key(requirement["source_slice_snssai"]) and selected not in authorized:
                return True
        return False

    @classmethod
    def _validate_entitlement_limited_qos_delivery(
        cls,
        optimizer_preview: Dict[str, Any],
        *,
        planning_request: PlanningRequest,
    ) -> List[str]:
        """Allow only QoS tuning that remains on a subscribed serving slice."""
        errors: List[str] = []
        authorized = {
            key
            for key in (
                cls._normalize_snssai_key(item)
                for item in planning_request.grounding_decision.slice_migration_authorization.authorized_snssais
            )
            if key
        }
        for requirement in cls._required_slice_change_constraints(planning_request):
            flow_id = requirement["flow_id"]
            source = cls._normalize_snssai_key(requirement["source_slice_snssai"])
            selected = cls._normalize_snssai_key(
                cls._extract_selected_slice(optimizer_preview, flow_id=flow_id)
            )
            if not selected:
                errors.append(
                    f"entitlement-limited QoS delivery requires a grounded serving-slice assignment for flow_id={flow_id}"
                )
            elif selected != source:
                errors.append(
                    "entitlement-limited QoS delivery must preserve the serving slice: "
                    f"flow_id={flow_id}, source_slice_snssai={source}, selected_slice={selected}"
                )
            elif authorized and selected not in authorized:
                errors.append(
                    "entitlement-limited QoS delivery selected a serving slice absent from subscription entitlement evidence: "
                    f"flow_id={flow_id}, selected_slice={selected}"
                )
        return errors

    @classmethod
    def _validate_am_targets_for_slice_migration(
        cls,
        *,
        planning_request: PlanningRequest,
        optimizer_preview: Dict[str, Any],
        am_policy: Any,
    ) -> List[str]:
        target_keys = {
            key
            for key in (
                cls._normalize_snssai_key(item.model_dump(mode="json") if hasattr(item, "model_dump") else item)
                for item in (getattr(am_policy, "target_snssais", None) or [])
            )
            if key
        }
        errors: List[str] = []
        for requirement in cls._required_slice_change_constraints(planning_request):
            selected = cls._normalize_snssai_key(cls._extract_selected_slice(optimizer_preview, flow_id=requirement["flow_id"]))
            source = cls._normalize_snssai_key(requirement["source_slice_snssai"])
            if selected and selected != source and selected not in target_keys:
                errors.append(
                    "AM target_snssais must include the optimizer-selected slice "
                    f"for flow_id={requirement['flow_id']}: {selected}"
                )
        return errors

    @staticmethod
    def _latest_optimizer_preview(planning_tool_evidence: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(planning_tool_evidence.get("latest_optimizer_preview") or {})
        summary = payload.get("summary")
        result = payload.get("result")
        if isinstance(result, dict):
            merged = dict(result)
            if isinstance(summary, dict):
                for key in ("status", "qos_meta_status", "qos_flow_assignments", "infeasible_reasons"):
                    if key in summary and key not in merged:
                        merged[key] = summary[key]
                if "qos_flow_assignments" in summary:
                    merged["qos_flow_assignments"] = summary["qos_flow_assignments"]
            return merged
        if isinstance(summary, dict):
            return dict(summary)
        if isinstance(payload, dict) and payload.get("status") is not None:
            return payload
        return {}

    @staticmethod
    def _latest_mobility_context(planning_tool_evidence: Dict[str, Any]) -> Dict[str, Any]:
        return dict(planning_tool_evidence.get("latest_mobility_context") or {})

    @staticmethod
    def _validate_sm_policy_target_bindings(
        sm_policies: List[Any],
        *,
        planning_request: PlanningRequest,
    ) -> List[str]:
        allowed_flow_app_ids = {
            str(flow.flow_id or "").strip(): _normalize_app_id(flow.app_id or "")
            for flow in (planning_request.grounding_decision.flows or [])
            if str(flow.flow_id or "").strip()
        }
        if not allowed_flow_app_ids:
            return []
        errors: List[str] = []
        for index, spec in enumerate(sm_policies or []):
            flow_id = str(getattr(spec, "flow_id", "") or "").strip()
            app_id = _normalize_app_id(getattr(spec, "app_id", "") or "")
            expected_app_id = allowed_flow_app_ids.get(flow_id)
            if expected_app_id is None:
                errors.append(
                    f"sm_policies[{index}].flow_id={flow_id or '<empty>'} is outside GroundingDecision flows "
                    f"{sorted(allowed_flow_app_ids)}"
                )
                continue
            if expected_app_id and app_id != expected_app_id:
                errors.append(
                    f"sm_policies[{index}].app_id must preserve GroundingDecision app_id={expected_app_id} "
                    f"for flow_id={flow_id}; got {app_id or '<empty>'}"
                )
        return errors

    @staticmethod
    def _validate_optimizer_preview(optimizer_preview: Any) -> None:
        if not isinstance(optimizer_preview, dict) or not optimizer_preview:
            raise ValueError("missing grounded optimizer preview")
        result_status = str(optimizer_preview.get("status") or "").strip().lower()
        infeasible_reasons = list(optimizer_preview.get("infeasible_reasons", []) or [])
        qos_plan = optimizer_preview.get("qos_plan") if isinstance(optimizer_preview, dict) else {}
        qos_meta = qos_plan.get("meta") if isinstance(qos_plan, dict) else {}
        qos_meta_status = str(optimizer_preview.get("qos_meta_status") or (qos_meta or {}).get("status") or "").strip()
        if result_status == DomainStatus.INCOMPLETE_CONTEXT.value:
            reason_text = "; ".join(str(item) for item in infeasible_reasons) or "missing required planning context"
            raise ValueError(f"incomplete_context: {reason_text}")
        if infeasible_reasons:
            raise ValueError("Joint optimizer returned infeasible result: " + "; ".join(str(item) for item in infeasible_reasons))
        if "infeasible" in qos_meta_status.lower():
            raise ValueError(f"Joint optimizer returned infeasible QoS plan: {qos_meta_status}")

    def _validate_optimizer_preview_payload(self, optimizer_preview: Dict[str, Any]) -> List[str]:
        try:
            self._validate_optimizer_preview(optimizer_preview)
        except Exception as exc:
            return [str(exc)]
        return []

    def _optimizer_preview_has_grounded_qos_assignments(self, optimizer_preview: Dict[str, Any], *, flow_ids: set[str]) -> bool:
        if not flow_ids or not optimizer_preview:
            return False
        result_status = str(optimizer_preview.get("status") or "").strip().lower()
        if result_status in {
            DomainStatus.REJECTED.value,
            DomainStatus.INCOMPLETE_CONTEXT.value,
            DomainStatus.FAILED.value,
        }:
            return False
        if self._validate_optimizer_preview_payload(optimizer_preview):
            return False
        return all(self._extract_qos_resource_keys(optimizer_preview, flow_id=flow_id) for flow_id in flow_ids)

    def _extract_qos_resource_keys(self, joint_result: Any, *, flow_id: str) -> List[str]:
        qos_plan = joint_result.get("qos_plan", {}) if isinstance(joint_result, dict) else {}
        flow_sets: List[List[Dict[str, Any]]] = []
        if isinstance(qos_plan, dict):
            target_apps = qos_plan.get("target_apps")
            if isinstance(target_apps, list):
                for item in target_apps:
                    if not isinstance(item, dict):
                        continue
                    flows = item.get("flows")
                    if isinstance(flows, list):
                        flow_sets.append(flows)
            target_app = qos_plan.get("target_app")
            if isinstance(target_app, dict):
                flows = target_app.get("flows")
                if isinstance(flows, list):
                    flow_sets.append(flows)

        keys: List[str] = []
        for flows in flow_sets:
            for item in flows:
                if not isinstance(item, dict):
                    continue
                if str(item.get("id") or "").strip() != flow_id:
                    continue
                allocation = item.get("allocation") if isinstance(item.get("allocation"), dict) else {}
                selected_slice = str(allocation.get("current_slice_snssai") or "").strip()
                if selected_slice:
                    keys.append(f"slice:{selected_slice}")
                    snssai = build_slice_snssai(selected_slice)
                    if snssai is not None:
                        keys.append(f"snssai:{json.dumps(snssai, sort_keys=True, ensure_ascii=False)}")
        summary_assignments = joint_result.get("qos_flow_assignments") if isinstance(joint_result, dict) else []
        if isinstance(summary_assignments, list):
            for item in summary_assignments:
                if not isinstance(item, dict):
                    continue
                if str(item.get("flow_id") or item.get("id") or "").strip() != flow_id:
                    continue
                selected_slice = str(
                    item.get("new_slice")
                    or item.get("current_slice_snssai")
                    or item.get("slice_snssai")
                    or ""
                ).strip()
                if selected_slice:
                    keys.append(f"slice:{selected_slice}")
                    snssai = build_slice_snssai(selected_slice)
                    if snssai is not None:
                        keys.append(f"snssai:{json.dumps(snssai, sort_keys=True, ensure_ascii=False)}")
        return list(dict.fromkeys(keys))

    def _validate_required_slice_changes(
        self,
        optimizer_preview: Dict[str, Any],
        *,
        planning_request: PlanningRequest,
    ) -> List[str]:
        errors: List[str] = []
        for requirement in self._required_slice_change_constraints(planning_request):
            flow_id = requirement["flow_id"]
            source_slice = requirement["source_slice_snssai"]
            selected_slice = self._extract_selected_slice(optimizer_preview, flow_id=flow_id)
            if not selected_slice:
                continue
            if selected_slice == source_slice:
                errors.append(
                    "required QoS slice migration was not satisfied: "
                    f"flow_id={flow_id} remained on source_slice_snssai={source_slice}"
                )
        return errors

    @staticmethod
    def _required_slice_change_constraints(planning_request: PlanningRequest) -> List[Dict[str, str]]:
        flow_source_slice = {
            str(flow.flow_id or "").strip(): str(flow.current_slice_snssai or "").strip()
            for flow in (planning_request.grounding_decision.flows or [])
            if str(flow.flow_id or "").strip()
        }
        requirements: List[Dict[str, str]] = []
        for constraint in planning_request.grounding_decision.qos_operation_constraints or []:
            if not constraint.require_slice_change:
                continue
            flow_id = str(constraint.flow_id or "").strip()
            source_slice = str(constraint.source_slice_snssai or flow_source_slice.get(flow_id) or "").strip()
            if flow_id and source_slice:
                requirements.append({"flow_id": flow_id, "source_slice_snssai": source_slice})

        # Global constraints are Main-level routing cues. They do not identify
        # a resolved flow or override IEA's explicit per-flow decision about a
        # serving-S-NSSAI migration.
        return requirements

    @staticmethod
    def _extract_selected_slice(joint_result: Any, *, flow_id: str) -> str:
        qos_plan = joint_result.get("qos_plan", {}) if isinstance(joint_result, dict) else {}
        flow_sets: List[List[Dict[str, Any]]] = []
        if isinstance(qos_plan, dict):
            for key in ("target_apps",):
                target_apps = qos_plan.get(key)
                if not isinstance(target_apps, list):
                    continue
                for item in target_apps:
                    if isinstance(item, dict) and isinstance(item.get("flows"), list):
                        flow_sets.append(item["flows"])
            target_app = qos_plan.get("target_app")
            if isinstance(target_app, dict) and isinstance(target_app.get("flows"), list):
                flow_sets.append(target_app["flows"])
        for flows in flow_sets:
            for item in flows:
                if not isinstance(item, dict):
                    continue
                if str(item.get("id") or "").strip() != flow_id:
                    continue
                allocation = item.get("allocation") if isinstance(item.get("allocation"), dict) else {}
                selected = str(allocation.get("current_slice_snssai") or "").strip()
                if selected:
                    return selected
        summary_assignments = joint_result.get("qos_flow_assignments") if isinstance(joint_result, dict) else []
        if isinstance(summary_assignments, list):
            for item in summary_assignments:
                if not isinstance(item, dict):
                    continue
                if str(item.get("flow_id") or item.get("id") or "").strip() != flow_id:
                    continue
                selected = str(item.get("new_slice") or item.get("current_slice_snssai") or item.get("slice_snssai") or "").strip()
                if selected:
                    return selected
        return ""

    @staticmethod
    def _preserved_app_id(planning_request: PlanningRequest) -> str:
        for flow in planning_request.grounding_decision.flows or []:
            app_id = _normalize_app_id(flow.app_id or "")
            if app_id:
                return app_id
        return ""

    @staticmethod
    def _preserved_flow_ids(planning_request: PlanningRequest) -> set[str]:
        return {
            str(flow.flow_id or "").strip()
            for flow in (planning_request.grounding_decision.flows or [])
            if str(flow.flow_id or "").strip()
        }

    @staticmethod
    def _preserved_flow_app_ids(planning_request: PlanningRequest) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for flow in planning_request.grounding_decision.flows or []:
            flow_id = str(flow.flow_id or "").strip()
            app_id = _normalize_app_id(flow.app_id or "")
            if flow_id and app_id:
                mapping[flow_id] = app_id
        return mapping

    @staticmethod
    def _preserved_association_id(planning_request: PlanningRequest) -> str:
        mobility_targets = planning_request.grounding_decision.grounding_evidence.grounded_mobility_targets
        if isinstance(mobility_targets, dict):
            summary = mobility_targets.get("summary")
            if isinstance(summary, dict):
                return str(summary.get("current_association_id") or "").strip()
        return ""


class PlanningArtifactValidator:
    def validate_compiled_plan(self, policy_plan: PolicyPlanDraft, planning_request: PlanningRequest) -> None:
        if policy_plan.planning_status == "needs_upstream_reground":
            if policy_plan.all_policies:
                raise ValueError("needs_upstream_reground must not contain executable policies")
            if not policy_plan.upstream_requests and not policy_plan.missing_evidence and not policy_plan.blocked_targets:
                raise ValueError("needs_upstream_reground must preserve explicit upstream requests or missing evidence")
            return
        if policy_plan.planning_status == "partial_plan":
            if not policy_plan.all_policies and not policy_plan.partial_policies:
                raise ValueError("partial_plan must preserve executable fragments or partial policies")
            if not policy_plan.missing_evidence and not policy_plan.blocked_targets and not policy_plan.planner_conflicts:
                raise ValueError("partial_plan must preserve unresolved gaps")
            return
        if not policy_plan.all_policies:
            raise ValueError("OptimizationStrategyAgent produced no policies for the requested domain.")

        active_domains = {
            str(item.value if hasattr(item, "value") else item or "").strip().lower()
            for item in (planning_request.context.shared_context.main_intent.requested_domains or [])
            if str(item.value if hasattr(item, "value") else item or "").strip()
        }
        if ControlDomain.QOS.value in active_domains:
            has_sm_policy = any(item.policy_type == "SmPolicyDecision" for item in policy_plan.all_policies)
            if not has_sm_policy:
                raise ValueError("OptimizationStrategyAgent did not include an executable SM policy for a qos-active round.")
        if ControlDomain.MOBILITY.value in active_domains:
            has_am_policy = any(item.policy_type == "PcfAmPolicyControlPolicyAssociation" for item in policy_plan.all_policies)
            if not has_am_policy:
                raise ValueError("OptimizationStrategyAgent did not include an executable AM policy for a mobility-active round.")
        retry_scope = _main_retry_scope(planning_request)
        if retry_scope == "target_stable":
            preserved_app_id = PlanningAdvisorValidator._preserved_app_id(planning_request)
            preserved_flow_ids = PlanningAdvisorValidator._preserved_flow_ids(planning_request)
            preserved_flow_app_ids = PlanningAdvisorValidator._preserved_flow_app_ids(planning_request)
            preserved_association_id = PlanningAdvisorValidator._preserved_association_id(planning_request)
            if preserved_flow_app_ids:
                drifted_policy_ids = [
                    item.policy_id
                    for item in policy_plan.all_policies
                    if item.policy_type == "SmPolicyDecision"
                    and str(item.flow_id or "").strip() in preserved_flow_app_ids
                    and _normalize_app_id(item.app_id) != preserved_flow_app_ids[str(item.flow_id or "").strip()]
                ]
                if drifted_policy_ids:
                    raise ValueError(
                        "target-stable retry compiled SM policies changed app_id away from preserved "
                        f"flow/app bindings: {drifted_policy_ids}"
                    )
            elif preserved_app_id:
                drifted_policy_ids = [
                    item.policy_id
                    for item in policy_plan.all_policies
                    if item.policy_type == "SmPolicyDecision" and _normalize_app_id(item.app_id) != preserved_app_id
                ]
                if drifted_policy_ids:
                    raise ValueError(
                        f"target-stable retry compiled SM policies changed app_id away from preserved {preserved_app_id}: {drifted_policy_ids}"
                    )
            if preserved_flow_ids:
                compiled_flow_ids = {
                    str(item.flow_id or "").strip()
                    for item in policy_plan.all_policies
                    if item.policy_type == "SmPolicyDecision" and str(item.flow_id or "").strip()
                }
                if compiled_flow_ids != preserved_flow_ids:
                    raise ValueError(
                        "target-stable retry compiled SM policies changed flow_ids away from preserved "
                        f"{sorted(preserved_flow_ids)} to {sorted(compiled_flow_ids)}"
                    )
            if preserved_association_id:
                compiled_association_ids = [
                    item.policy_id
                    for item in policy_plan.all_policies
                    if item.policy_type == "PcfAmPolicyControlPolicyAssociation"
                ]
                if compiled_association_ids and any(item != preserved_association_id for item in compiled_association_ids):
                    raise ValueError(
                        "target-stable retry compiled AM policy changed association_id away from preserved "
                        f"{preserved_association_id}: {compiled_association_ids}"
                    )
        if ControlDomain.QOS.value in active_domains and ControlDomain.MOBILITY.value in active_domains:
            self._validate_joint_snssai_consistency(policy_plan)

    @staticmethod
    def _validate_joint_snssai_consistency(policy_plan: PolicyPlanDraft) -> None:
        selected_snssais: set[str] = set()
        sm_policy_count = 0
        allowed_snssais: set[str] = set()
        for item in policy_plan.all_policies:
            if item.policy_type == "SmPolicyDecision":
                sm_policy_count += 1
                for resource_key in item.resource_keys or []:
                    if str(resource_key).startswith("snssai:"):
                        selected_snssais.add(str(resource_key))
            elif item.policy_type == "PcfAmPolicyControlPolicyAssociation":
                request = item.policy_details.get("request") if isinstance(item.policy_details, dict) else {}
                for snssai in request.get("allowedSnssais") or []:
                    allowed_snssais.add(f"snssai:{json.dumps(snssai, sort_keys=True, ensure_ascii=False)}")
        if sm_policy_count and not selected_snssais:
            raise ValueError("joint qos/mobility plan is missing optimizer-backed S-NSSAI resource keys for SM policies")
        uncovered = sorted(selected_snssais - allowed_snssais)
        if uncovered:
            raise ValueError(
                "QoS-selected S-NSSAI values are not covered by mobility allowedSnssais: "
                + ", ".join(uncovered)
            )


__all__ = ["PlanningAdvisorValidator", "PlanningArtifactValidator"]
