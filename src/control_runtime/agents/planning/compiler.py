from __future__ import annotations

import json
from typing import Any, Dict, List

from ..grounding.contracts import FlowSelector
from .contracts import PlanningRationale, PlanningRequest, PolicyDraft, PolicyPlanDraft
from .evidence import build_planning_evidence
from .planning_validation import (
    PlanningArtifactValidator,
    PlanningAdvisorValidator,
    json_friendly as _json_friendly,
    normalize_app_id as _normalize_app_id,
    normalize_policy_plan_draft,
)
from .response_models import (
    AmPolicySpec,
    OsaAdvisorOutput,
    PartialPolicySpec,
    SmPolicySpec,
    SnssaiSpec,
    UrspPolicySpec,
    decode_snssai_key,
)



def _build_slice_snssai(slice_code: str) -> Dict[str, Any] | None:
    try:
        return decode_snssai_key(slice_code).model_dump(mode="json")
    except ValueError:
        return None


def _planning_supi(planning_request: PlanningRequest) -> str:
    main_supi = planning_request.context.shared_context.main_intent.scope.supi or ""
    if main_supi:
        return main_supi
    return next(
        (
            str(flow.supi or "").strip()
            for flow in (planning_request.grounding_decision.flows or [])
            if str(flow.supi or "").strip()
        ),
        "",
    )


def _main_retry_scope(planning_request: PlanningRequest) -> str:
    retry_scope = planning_request.context.shared_context.main_intent.retry_scope
    return str(getattr(retry_scope, "value", retry_scope) or "").strip().lower()


class OptimizationStrategyCompiler:
    """Compile a validated OSA response into the canonical PolicyPlanDraft."""

    def __init__(self) -> None:
        self.advisor_validator = PlanningAdvisorValidator()
        self.plan_validator = PlanningArtifactValidator()
        self.validator = self.plan_validator

    @staticmethod
    def build_planning_evidence(planning_request: PlanningRequest) -> Dict[str, Any]:
        return build_planning_evidence(planning_request)

    def validate_advisor_output(
        self,
        *,
        advisor_output: OsaAdvisorOutput,
        planning_request: PlanningRequest,
        grounding_tools: List[str],
        planning_tool_evidence: Dict[str, Any] | None = None,
    ) -> List[str]:
        return self.advisor_validator.validate_advisor_output(
            advisor_output=advisor_output,
            planning_request=planning_request,
            grounding_tools=grounding_tools,
            planning_tool_evidence=planning_tool_evidence,
        )

    def assemble_policy_plan(
        self,
        *,
        advisor_output: OsaAdvisorOutput,
        planning_request: PlanningRequest,
        planning_tool_evidence: Dict[str, Any],
    ) -> PolicyPlanDraft:
        planning_status = str(advisor_output.planning_status or "").strip().lower()
        if planning_status == "needs_upstream_reground":
            return self.build_upstream_reground_plan(
                planning_request=planning_request,
                reason="; ".join(advisor_output.upstream_requests or advisor_output.missing_evidence or advisor_output.blocked_targets),
                advisor_output=advisor_output,
            )
        optimizer_preview = PlanningAdvisorValidator._latest_optimizer_preview(planning_tool_evidence)
        mobility_context = PlanningAdvisorValidator._latest_mobility_context(planning_tool_evidence)
        if planning_status == "executable_plan" and advisor_output.sm_policies:
            PlanningAdvisorValidator._validate_optimizer_preview(optimizer_preview)
        if planning_status == "executable_plan" and advisor_output.am_policy is not None and not mobility_context:
            raise ValueError("am_policy compilation requires grounded mobility context")

        optimizer_result = {
            "objective_breakdown": dict(optimizer_preview.get("objective_breakdown") or {}) if isinstance(optimizer_preview, dict) else {},
            "cross_domain_verdicts": [
                _json_friendly(item)
                for item in ((optimizer_preview.get("cross_domain_verdicts") if isinstance(optimizer_preview, dict) else []) or [])
            ],
        }

        plan = PolicyPlanDraft(
            supi=_planning_supi(planning_request),
            session_id=planning_request.context.shared_context.main_intent.scope.session_id,
            snapshot_id=planning_request.context.shared_context.main_intent.scope.snapshot_id,
            planning_status=planning_status,
            optimizer_result=optimizer_result,
            planning_rationale=PlanningRationale(
                selected_strategy_profile=str(
                    planning_request.context.shared_context.main_intent.objective_profile.profile_name
                    or ""
                ).strip(),
                decisive_evidence=[
                    item
                    for item in [
                        "target_binding_reused" if _main_retry_scope(planning_request) == "target_stable" else "",
                        "tool:preview_qos_optimizer" if advisor_output.sm_policies else "",
                        "tool:inspect_mobility_ue_policies" if advisor_output.am_policy is not None else "",
                        "mediator_constraints" if planning_request.context.unified_constraints else "",
                        "revision_requests" if planning_request.context.revision_requests else "",
                    ]
                    if item
                ],
                active_constraints=[
                    str(item)
                    for item in self._normalized_hard_constraints(
                        planning_request.context.unified_constraints
                    )
                    if str(item).strip()
                ],
                explanation=str(advisor_output.rationale or "").strip(),
                rejected_alternatives=[
                    item
                    for item in [
                        "target_rebinding" if _main_retry_scope(planning_request) == "target_stable" else "",
                    ]
                    if item
                ],
                main_constraints=[
                    str(item)
                    for item in planning_request.context.shared_context.main_intent.required_evidence
                    if str(item).strip()
                ],
                iea_grounding_basis=[
                    str(item)
                    for item in (
                        list((planning_request.grounding_decision.grounding_evidence.evidence_sources or {}).keys())
                        + [
                            str(item.get("flow_id") or "").strip()
                            for item in (planning_request.grounding_decision.grounding_evidence.grounded_flows or [])
                            if str(item.get("flow_id") or "").strip()
                        ]
                    )
                    if str(item).strip()
                ],
                osa_decision_basis=[
                    str(item)
                    for item in [
                        *advisor_output.missing_evidence,
                        *advisor_output.blocked_targets,
                        *advisor_output.upstream_requests,
                        *advisor_output.planner_conflicts,
                    ]
                    if str(item).strip()
                ],
                unresolved_gaps=[str(item) for item in (advisor_output.missing_evidence or []) if str(item).strip()],
            ),
            all_policies=[],
            partial_policies=[],
            missing_evidence=[str(item) for item in (advisor_output.missing_evidence or []) if str(item).strip()],
            blocked_targets=[str(item) for item in (advisor_output.blocked_targets or []) if str(item).strip()],
            upstream_requests=[str(item) for item in (advisor_output.upstream_requests or []) if str(item).strip()],
            planner_conflicts=[str(item) for item in (advisor_output.planner_conflicts or []) if str(item).strip()],
        )
        if planning_status == "partial_plan":
            plan.partial_policies = self._build_partial_policy_drafts(
                partial_policies=advisor_output.partial_policies,
                planning_request=planning_request,
            )

        # The optimizer, rather than the advisor, is authoritative for the
        # S-NSSAI actually selected for each QoS flow.  Gather it once here
        # and feed the exact same assignments into both the SM payloads and
        # the AM policy compiler.  Previously the advisor could omit an
        # optimizer-selected S-NSSAI from allowed_snssais, causing a joint
        # QoS/mobility plan to fail only after it had otherwise compiled.
        sm_assignments: List[tuple[SmPolicySpec, List[str], str]] = []
        for spec in advisor_output.sm_policies:
            resource_keys = PlanningAdvisorValidator()._extract_qos_resource_keys(
                optimizer_preview,
                flow_id=spec.flow_id,
            )
            if not resource_keys:
                raise ValueError(
                    f"optimizer preview does not contain a grounded QoS assignment for flow_id={spec.flow_id}"
                )
            sm_assignments.append(
                (
                    self._with_optimizer_bandwidths(spec, optimizer_preview),
                    resource_keys,
                    self._target_slice_snssai(resource_keys=resource_keys, flow_id=spec.flow_id),
                )
            )

        if advisor_output.am_policy is not None:
            plan.all_policies.append(
                PolicyDraft(
                    recommended_actions=[],
                    supi=_planning_supi(planning_request),
                    app_id="",
                    flow_id=None,
                    target_type="ue",
                    policy_id=self._resolve_am_association_id(
                        planning_request=planning_request,
                        mobility_context=mobility_context,
                    ),
                    policy_type="PcfAmPolicyControlPolicyAssociation",
                    policy_details=self._build_am_policy_details(
                        advisor_output.am_policy,
                        planning_request=planning_request,
                        mobility_context=mobility_context,
                        qos_selected_snssais=[assignment[2] for assignment in sm_assignments],
                    ),
                )
            )

        for spec, resource_keys, target_slice_snssai in sm_assignments:
            flow_ctx = self._resolve_flow(planning_request, spec.flow_id)
            if _normalize_app_id(spec.app_id) != _normalize_app_id(flow_ctx.app_id or ""):
                raise ValueError(f"sm policy app_id does not match resolved flow context for flow_id={spec.flow_id}")
            plan.all_policies.append(
                PolicyDraft(
                    recommended_actions=[],
                    supi=str(flow_ctx.supi or _planning_supi(planning_request)).strip(),
                    app_id=_normalize_app_id(spec.app_id),
                    flow_id=spec.flow_id,
                    target_type="flow",
                    policy_id=f"smp-{_normalize_app_id(spec.app_id)}-{spec.flow_id}",
                    policy_type="SmPolicyDecision",
                    resource_keys=resource_keys,
                    policy_details=self._build_sm_policy_details(
                        spec,
                        flow_ctx,
                        target_slice_snssai=target_slice_snssai,
                    ),
                )
            )

        for index, spec in enumerate(advisor_output.ursp_policies, start=1):
            flow_ctx = self._resolve_flow(planning_request, spec.flow_id) if spec.flow_id else None
            if spec.target_type == "flow" and flow_ctx is None:
                raise ValueError(f"flow-scoped URSP policy references unknown flow_id={spec.flow_id}")
            if spec.target_type == "flow" and _normalize_app_id(spec.app_id) != _normalize_app_id(flow_ctx.app_id or ""):
                raise ValueError(f"ursp policy app_id does not match resolved flow context for flow_id={spec.flow_id}")
            plan.all_policies.append(
                PolicyDraft(
                    recommended_actions=[],
                    supi=str((flow_ctx.supi if flow_ctx is not None else "") or _planning_supi(planning_request)).strip(),
                    app_id=_normalize_app_id(spec.app_id),
                    flow_id=spec.flow_id,
                    target_type=spec.target_type,
                    policy_id=f"ursp-{_normalize_app_id(spec.app_id)}-{spec.flow_id or index}",
                    policy_type="UrspRuleRequest",
                    policy_details=self._build_ursp_policy_details(spec),
                )
            )

        normalized = normalize_policy_plan_draft(plan)
        self.validator.validate_compiled_plan(normalized, planning_request)
        return normalized

    @staticmethod
    def _normalized_hard_constraints(unified_constraints: Any) -> List[Any]:
        if not isinstance(unified_constraints, dict):
            return []
        raw_constraints = unified_constraints.get("hard_constraints")
        if raw_constraints is None:
            return []
        if not isinstance(raw_constraints, list):
            raise TypeError("planning_request.context.unified_constraints.hard_constraints must be a list when present")
        return raw_constraints

    @staticmethod
    def _resolve_flow(planning_request: PlanningRequest, flow_id: str) -> FlowSelector:
        target = str(flow_id or "").strip()
        for flow in planning_request.grounding_decision.flows:
            if str(flow.flow_id or "").strip() == target:
                return flow
        raise ValueError(f"unknown flow_id in advisor output: {flow_id}")

    @staticmethod
    def _target_slice_snssai(*, resource_keys: List[str], flow_id: str) -> str:
        selected_slices = {
            str(resource_key).removeprefix("slice:").strip()
            for resource_key in resource_keys
            if str(resource_key).startswith("slice:") and str(resource_key).removeprefix("slice:").strip()
        }
        if len(selected_slices) != 1:
            raise ValueError(
                f"optimizer preview must provide exactly one target slice for flow_id={flow_id}; "
                f"received {sorted(selected_slices)}"
            )
        target_slice_snssai = selected_slices.pop()
        if _build_slice_snssai(target_slice_snssai) is None:
            raise ValueError(
                f"optimizer preview returned invalid target slice S-NSSAI for flow_id={flow_id}: {target_slice_snssai}"
            )
        return target_slice_snssai

    @staticmethod
    def _with_optimizer_bandwidths(spec: SmPolicySpec, optimizer_preview: Any) -> SmPolicySpec:
        """Make the optimizer allocation authoritative without changing absent legacy evidence."""
        assignments = optimizer_preview.get("qos_flow_assignments") if isinstance(optimizer_preview, dict) else []
        assignment = next(
            (
                item
                for item in (assignments or [])
                if isinstance(item, dict) and str(item.get("flow_id") or "").strip() == spec.flow_id
            ),
            None,
        )
        if not assignment:
            return spec
        try:
            bandwidth_ul = float(assignment["bw_ul"])
            bandwidth_dl = float(assignment["bw_dl"])
        except (KeyError, TypeError, ValueError):
            return spec
        if bandwidth_ul <= 0 or bandwidth_dl <= 0:
            return spec
        updates: Dict[str, Any] = {
            "max_br_ul_mbps": bandwidth_ul,
            "max_br_dl_mbps": bandwidth_dl,
        }
        if spec.gbr_ul_mbps is not None:
            updates["gbr_ul_mbps"] = min(float(spec.gbr_ul_mbps), bandwidth_ul)
        if spec.gbr_dl_mbps is not None:
            updates["gbr_dl_mbps"] = min(float(spec.gbr_dl_mbps), bandwidth_dl)
        return spec.model_copy(update=updates)

    @staticmethod
    def _build_sm_policy_details(
        spec: SmPolicySpec,
        flow_ctx: FlowSelector,
        *,
        target_slice_snssai: str,
    ) -> Dict[str, Any]:
        flow_id = str(spec.flow_id or "").strip()
        if not flow_id:
            raise ValueError("SmPolicySpec requires flow_id")
        app_id = _normalize_app_id(spec.app_id)
        if not app_id:
            raise ValueError(f"SmPolicySpec requires app_id for flow_id={flow_id}")
        pcc_id = f"pcc-{flow_id}"
        qos_id = f"qos-{flow_id}"
        gbr_ul_mbps = OptimizationStrategyCompiler._bounded_gbr_mbps(
            spec.gbr_ul_mbps,
            spec.max_br_ul_mbps,
        )
        gbr_dl_mbps = OptimizationStrategyCompiler._bounded_gbr_mbps(
            spec.gbr_dl_mbps,
            spec.max_br_dl_mbps,
        )
        qos_payload: Dict[str, Any] = {
            "qosId": qos_id,
            "5qi": OptimizationStrategyCompiler._map_5qi_by_service_type(flow_ctx.service_type_id),
            "priorityLevel": int(spec.priority),
            "packetDelayBudget": int(spec.target_latency_ms),
            "packetErrorRate": str(spec.packet_error_rate),
            "maxbrUl": str(spec.max_br_ul_mbps),
            "maxbrDl": str(spec.max_br_dl_mbps),
        }
        if gbr_ul_mbps is not None:
            qos_payload["gbrUl"] = str(gbr_ul_mbps)
        if gbr_dl_mbps is not None:
            qos_payload["gbrDl"] = str(gbr_dl_mbps)
        if spec.target_jitter_ms is not None:
            qos_payload["jitterReq"] = spec.target_jitter_ms
        target_slice = _build_slice_snssai(target_slice_snssai)
        if target_slice is None:
            raise ValueError(f"invalid target slice S-NSSAI for flow_id={flow_id}: {target_slice_snssai}")
        return {
            "_preserve_explicit_qos_values": True,
            "upstreamSmPolicyContextData": {"sliceInfo": target_slice},
            "pccRules": {
                pcc_id: {
                    "pccRuleId": pcc_id,
                    "precedence": int(spec.priority),
                    "refQosData": [qos_id],
                    "appId": app_id,
                    "flowInfos": [
                        {
                            "flowDirection": "BIDIRECTIONAL",
                            "flowDescription": str(
                                spec.flow_description
                                or flow_ctx.description
                                or flow_ctx.name
                                or flow_id
                            ),
                        }
                    ],
                }
            },
            "qosDecs": {qos_id: qos_payload},
        }

    @staticmethod
    def _bounded_gbr_mbps(gbr_mbps: float | None, max_br_mbps: float) -> float | None:
        if gbr_mbps is None:
            return None
        return min(float(gbr_mbps), float(max_br_mbps))

    @staticmethod
    def _map_5qi_by_service_type(service_type_id: int | None) -> int:
        try:
            service_id = int(service_type_id or 0)
        except (TypeError, ValueError):
            service_id = 0
        return {1: 9, 2: 7, 3: 65}.get(service_id, 9)

    def _resolve_am_association_id(self, *, planning_request: PlanningRequest, mobility_context: Dict[str, Any]) -> str:
        preserved = PlanningAdvisorValidator._preserved_association_id(planning_request)
        if preserved:
            return preserved
        mobility_summary = mobility_context.get("mobilitySummary") if isinstance(mobility_context, dict) else {}
        association_id = str((mobility_summary or {}).get("currentAssociationId") or "").strip()
        if not association_id:
            raise ValueError("mobility context does not contain a grounded association_id")
        return association_id

    def _build_am_policy_details(
        self,
        spec: AmPolicySpec,
        *,
        planning_request: PlanningRequest,
        mobility_context: Dict[str, Any],
        qos_selected_snssais: List[str] | None = None,
    ) -> Dict[str, Any]:
        am_policy_context = mobility_context.get("amPolicyContext") if isinstance(mobility_context, dict) else {}
        access_context = mobility_context.get("accessMobilityContext") if isinstance(mobility_context, dict) else {}
        if not isinstance(am_policy_context, dict) or not am_policy_context:
            raise ValueError("mobility context does not contain grounded amPolicyContext")
        association_id = self._resolve_am_association_id(
            planning_request=planning_request,
            mobility_context=mobility_context,
        )
        association_payload = am_policy_context.get("associations", {}).get(association_id)
        if not isinstance(association_payload, dict) or not association_payload:
            raise ValueError(f"mobility context does not contain grounded association payload for {association_id}")
        request_payload = association_payload.get("request")
        policy_payload = association_payload
        if not isinstance(request_payload, dict) or not request_payload:
            raise ValueError("mobility context does not contain a grounded AM request payload")
        if not isinstance(policy_payload, dict) or not policy_payload:
            raise ValueError("mobility context does not contain a grounded AM policy payload")
        if not str(request_payload.get("notificationUri") or "").strip():
            raise ValueError("grounded AM request payload is missing notificationUri")

        request_payload = _json_friendly(request_payload)
        policy_payload = _json_friendly(policy_payload)
        request_payload["supi"] = _planning_supi(planning_request)
        selected_snssai_payloads: List[Dict[str, Any]] = []
        for snssai in qos_selected_snssais or []:
            selected_payload = _build_slice_snssai(snssai)
            if selected_payload is not None:
                selected_snssai_payloads.append(selected_payload)
        # An SM policy is generated exclusively from an optimizer-selected
        # slice.  If AM is active in the same plan, that slice must be allowed
        # and targeted by the AM association as well.  Preserve any separately
        # grounded AM choices, but add the code-grounded QoS selections so the
        # two policy domains cannot drift because of an advisor omission.
        request_payload["allowedSnssais"] = self._merge_snssai_payloads(
            [item.model_dump(mode="json") for item in spec.allowed_snssais],
            selected_snssai_payloads,
        )
        request_payload["targetSnssais"] = self._merge_snssai_payloads(
            [item.model_dump(mode="json") for item in spec.target_snssais],
            selected_snssai_payloads,
        )
        request_payload["rfsp"] = int(spec.rfsp)
        if spec.ue_ambr_ul_mbps is not None or spec.ue_ambr_dl_mbps is not None:
            if spec.ue_ambr_ul_mbps is None or spec.ue_ambr_dl_mbps is None:
                raise ValueError("ue_ambr_ul_mbps and ue_ambr_dl_mbps must be provided together")
            request_payload["ueAmbr"] = {
                "uplink": str(spec.ue_ambr_ul_mbps),
                "downlink": str(spec.ue_ambr_dl_mbps),
            }
        elif isinstance(access_context, dict) and access_context.get("ueAmbr"):
            request_payload["ueAmbr"] = _json_friendly(access_context.get("ueAmbr"))
        if request_payload.get("servAreaRes") is None and am_policy_context.get("servAreaRes") is not None:
            request_payload["servAreaRes"] = _json_friendly(am_policy_context.get("servAreaRes"))

        policy_payload["request"] = request_payload
        policy_payload["triggers"] = list(spec.triggers)
        policy_payload["rfsp"] = request_payload["rfsp"]
        if "ueAmbr" in request_payload:
            policy_payload["ueAmbr"] = request_payload["ueAmbr"]
        if "servAreaRes" in request_payload:
            policy_payload["servAreaRes"] = request_payload["servAreaRes"]
        return policy_payload

    @staticmethod
    def _merge_snssai_payloads(*groups: List[Dict[str, Any] | None]) -> List[Dict[str, Any]]:
        """Deduplicate S-NSSAI payloads without changing their PCF wire shape."""

        merged: Dict[str, Dict[str, Any]] = {}
        for group in groups:
            for payload in group:
                if not isinstance(payload, dict):
                    continue
                normalized = _json_friendly(payload)
                if not isinstance(normalized, dict):
                    continue
                key = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
                merged.setdefault(key, normalized)
        return list(merged.values())

    @staticmethod
    def _build_ursp_policy_details(spec: UrspPolicySpec) -> Dict[str, Any]:
        traffic_desc_payload: Dict[str, Any] = {}
        if spec.traffic_desc is not None:
            if spec.traffic_desc.flow_descs:
                traffic_desc_payload["flowDescs"] = list(spec.traffic_desc.flow_descs)
            if spec.traffic_desc.dnns:
                traffic_desc_payload["dnns"] = list(spec.traffic_desc.dnns)
            if spec.traffic_desc.app_ids:
                os_id = str(spec.traffic_desc.os_id or "default").strip() or "default"
                traffic_desc_payload["appDescs"] = {
                    os_id: {"osId": os_id, "appIds": list(spec.traffic_desc.app_ids)}
                }

        route_sets = []
        for item in spec.route_sel_param_sets:
            payload: Dict[str, Any] = {
                "dnn": item.dnn,
                "precedence": int(item.precedence or spec.relat_precedence),
            }
            if item.snssai is not None:
                payload["snssai"] = item.snssai.model_dump(mode="json")
            route_sets.append(payload)

        details: Dict[str, Any] = {
            "relatPrecedence": int(spec.relat_precedence),
            "routeSelParamSets": route_sets,
        }
        if traffic_desc_payload:
            details["trafficDesc"] = traffic_desc_payload
        return details

    @staticmethod
    def _build_partial_policy_drafts(
        *,
        partial_policies: list[PartialPolicySpec],
        planning_request: PlanningRequest,
    ) -> List[PolicyDraft]:
        drafts: List[PolicyDraft] = []
        for index, item in enumerate(partial_policies, start=1):
            drafts.append(
                PolicyDraft(
                    recommended_actions=[],
                    supi=_planning_supi(planning_request),
                    app_id=item.app_id,
                    flow_id=item.flow_id,
                    target_type=item.target_type,
                    policy_id=f"partial-{index}",
                    policy_type=item.policy_type,
                    resource_keys=item.resource_keys,
                    policy_details=item.policy_details,
                )
            )
        return drafts

    @staticmethod
    def build_upstream_reground_plan(
        *,
        planning_request: PlanningRequest,
        reason: str,
        advisor_output: OsaAdvisorOutput | None = None,
    ) -> PolicyPlanDraft:
        missing_evidence = list((advisor_output.missing_evidence if advisor_output is not None else []) or [])
        blocked_targets = list((advisor_output.blocked_targets if advisor_output is not None else []) or [])
        upstream_requests = list((advisor_output.upstream_requests if advisor_output is not None else []) or [])
        planner_conflicts = list((advisor_output.planner_conflicts if advisor_output is not None else []) or [])
        rationale = str((advisor_output.rationale if advisor_output is not None else "") or reason or "").strip()
        if not upstream_requests:
            upstream_requests = [reason] if str(reason or "").strip() else []
        return PolicyPlanDraft(
            supi=_planning_supi(planning_request),
            session_id=planning_request.context.shared_context.main_intent.scope.session_id,
            snapshot_id=planning_request.context.shared_context.main_intent.scope.snapshot_id,
            planning_status="needs_upstream_reground",
            planning_rationale=PlanningRationale(
                selected_strategy_profile=str(
                    planning_request.context.shared_context.main_intent.objective_profile.profile_name
                    or ""
                ).strip(),
                explanation=rationale,
                unresolved_gaps=[*missing_evidence, *blocked_targets, *upstream_requests],
                main_constraints=[
                    str(item)
                    for item in planning_request.context.shared_context.main_intent.required_evidence
                    if str(item).strip()
                ],
            ),
            missing_evidence=[str(item) for item in missing_evidence if str(item).strip()],
            blocked_targets=[str(item) for item in blocked_targets if str(item).strip()],
            upstream_requests=[str(item) for item in upstream_requests if str(item).strip()],
            planner_conflicts=[str(item) for item in planner_conflicts if str(item).strip()],
        )


__all__ = ["OptimizationStrategyCompiler"]
