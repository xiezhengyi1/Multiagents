from __future__ import annotations

import json
from typing import Any, Dict, List

from ...domain.collaboration import PlanningRequest
from ...domain.control_plane import ControlDomain, DomainStatus
from ...domain.policy_plan import PolicyPlanDraft
from ...context.evidence import build_slice_snssai
from .policy_normalizer import normalize_app_id as _normalize_app_id
from .response_models import OsaAdvisorOutput


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
            str(item).strip().lower()
            for item in (planning_request.context.active_domains or [])
            if str(item).strip()
        }
        normalized_tool_evidence = dict(planning_tool_evidence or {})
        retry_scope = str(planning_request.context.main_retry_scope or "").strip().lower()
        preserved_app_id = self._preserved_app_id(planning_request)
        preserved_flow_ids = self._preserved_flow_ids(planning_request)
        preserved_flow_app_ids = self._preserved_flow_app_ids(planning_request)
        planning_status = str(advisor_output.planning_status or "").strip().lower()

        has_sm = bool(advisor_output.sm_policies)
        has_am = advisor_output.am_policy is not None
        has_ursp = bool(advisor_output.ursp_policies)

        if has_sm and "qos" not in domains:
            errors.append("advisor emitted sm_policies outside qos-active planning")
        if has_am and "mobility" not in domains:
            errors.append("advisor emitted am_policy outside mobility-active planning")
        if planning_status == "executable_plan" and "qos" in domains and not has_sm:
            errors.append("qos-active planning requires sm_policies")
        if planning_status == "executable_plan" and "mobility" in domains and not has_am:
            errors.append("mobility-active planning requires am_policy")
        if has_sm:
            errors.extend(self._validate_sm_policy_qos_bounds(advisor_output.sm_policies))
        optimizer_preview = self._latest_optimizer_preview(normalized_tool_evidence)
        mobility_context = self._latest_mobility_context(normalized_tool_evidence)
        if (
            "qos" in domains
            and planning_status != "executable_plan"
            and self._optimizer_preview_has_grounded_qos_assignments(
                optimizer_preview,
                flow_ids=preserved_flow_ids,
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
    def _latest_optimizer_preview(planning_tool_evidence: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(planning_tool_evidence.get("latest_optimizer_preview") or {})
        result = payload.get("result")
        if isinstance(result, dict):
            return dict(result)
        summary = payload.get("summary")
        if isinstance(summary, dict):
            return dict(summary)
        if isinstance(payload, dict) and payload.get("status") is not None:
            return payload
        return {}

    @staticmethod
    def _latest_mobility_context(planning_tool_evidence: Dict[str, Any]) -> Dict[str, Any]:
        return dict(planning_tool_evidence.get("latest_mobility_context") or {})

    @staticmethod
    def _validate_sm_policy_qos_bounds(sm_policies: List[Any]) -> List[str]:
        errors: List[str] = []
        for index, spec in enumerate(sm_policies or []):
            max_ul = float(getattr(spec, "max_br_ul_mbps", 0.0) or 0.0)
            max_dl = float(getattr(spec, "max_br_dl_mbps", 0.0) or 0.0)
            gbr_ul = getattr(spec, "gbr_ul_mbps", None)
            gbr_dl = getattr(spec, "gbr_dl_mbps", None)
            if gbr_ul is not None and float(gbr_ul) > max_ul + 1e-9:
                errors.append(
                    f"sm_policies[{index}].gbr_ul_mbps must not exceed max_br_ul_mbps "
                    f"({float(gbr_ul)} > {max_ul})"
                )
            if gbr_dl is not None and float(gbr_dl) > max_dl + 1e-9:
                errors.append(
                    f"sm_policies[{index}].gbr_dl_mbps must not exceed max_br_dl_mbps "
                    f"({float(gbr_dl)} > {max_dl})"
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

    @staticmethod
    def _preserved_app_id(planning_request: PlanningRequest) -> str:
        return _normalize_app_id(planning_request.operation_intent.app_id or "")

    @staticmethod
    def _preserved_flow_ids(planning_request: PlanningRequest) -> set[str]:
        return {
            str(flow.flow_id or "").strip()
            for flow in (planning_request.operation_intent.flows or [])
            if str(flow.flow_id or "").strip()
        }

    @staticmethod
    def _preserved_flow_app_ids(planning_request: PlanningRequest) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for flow in planning_request.operation_intent.flows or []:
            flow_id = str(flow.flow_id or "").strip()
            app_id = _normalize_app_id(flow.app_id or "")
            if flow_id and app_id:
                mapping[flow_id] = app_id
        return mapping

    @staticmethod
    def _preserved_association_id(planning_request: PlanningRequest) -> str:
        mobility_targets = planning_request.operation_intent.grounding_evidence.grounded_mobility_targets
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
            str(item).strip().lower()
            for item in (planning_request.context.active_domains or [])
            if str(item).strip()
        }
        if ControlDomain.QOS.value in active_domains:
            has_sm_policy = any(item.policy_type == "SmPolicyDecision" for item in policy_plan.all_policies)
            if not has_sm_policy:
                raise ValueError("OptimizationStrategyAgent did not include an executable SM policy for a qos-active round.")
        if ControlDomain.MOBILITY.value in active_domains:
            has_am_policy = any(item.policy_type == "PcfAmPolicyControlPolicyAssociation" for item in policy_plan.all_policies)
            if not has_am_policy:
                raise ValueError("OptimizationStrategyAgent did not include an executable AM policy for a mobility-active round.")
        retry_scope = str(planning_request.context.main_retry_scope or "").strip().lower()
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
