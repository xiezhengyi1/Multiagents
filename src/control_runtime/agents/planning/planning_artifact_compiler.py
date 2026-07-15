from __future__ import annotations

from typing import Any, Dict, List

from ...domain.collaboration import PlanningRequest
from ...domain.policy_plan import FlowSelector, PlanningRationale, PolicyDraft, PolicyPlanDraft
from .planning_validation import (
    PlanningArtifactValidator,
    PlanningAdvisorValidator,
    json_friendly as _json_friendly,
    normalize_app_id as _normalize_app_id,
    normalize_policy_plan_draft,
)
from .response_models import AmPolicySpec, OsaAdvisorOutput, SmPolicySpec, UrspPolicySpec


class PlanningArtifactCompiler:
    def __init__(self, *, validator: PlanningArtifactValidator | None = None) -> None:
        self.validator = validator or PlanningArtifactValidator()

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
            "snapshot_writeback_patch": self._build_snapshot_writeback_patch(optimizer_preview),
        }

        plan = PolicyPlanDraft(
            supi=str(planning_request.operation_intent.supi or "").strip(),
            session_id=str(planning_request.context.session_id or "").strip(),
            snapshot_id=str(planning_request.context.snapshot_id or "").strip(),
            planning_status=planning_status,
            optimizer_result=optimizer_result,
            planning_rationale=PlanningRationale(
                selected_strategy_profile=str(
                    planning_request.context.shared_context.initial_intent.objective_profile.get("profile_name")
                    or ""
                ).strip(),
                decisive_evidence=[
                    item
                    for item in [
                        "target_binding_reused" if str(planning_request.context.retry_scope or "").strip().lower() == "target_stable" else "",
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
                        "target_rebinding" if str(planning_request.context.retry_scope or "").strip().lower() == "target_stable" else "",
                    ]
                    if item
                ],
                main_constraints=[
                    str(item)
                    for item in planning_request.context.shared_context.initial_intent.required_evidence
                    if str(item).strip()
                ],
                iea_grounding_basis=[
                    str(item)
                    for item in (
                        list((planning_request.operation_intent.grounding_evidence.evidence_sources or {}).keys())
                        + [
                            str(item.get("flow_id") or "").strip()
                            for item in (planning_request.operation_intent.grounding_evidence.grounded_flows or [])
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

        for spec in advisor_output.sm_policies:
            flow_ctx = self._resolve_flow(planning_request, spec.flow_id)
            if _normalize_app_id(spec.app_id) != _normalize_app_id(flow_ctx.app_id or ""):
                raise ValueError(f"sm policy app_id does not match resolved flow context for flow_id={spec.flow_id}")
            resource_keys = PlanningAdvisorValidator()._extract_qos_resource_keys(optimizer_preview, flow_id=spec.flow_id)
            if not resource_keys:
                raise ValueError(f"optimizer preview does not contain a grounded QoS assignment for flow_id={spec.flow_id}")
            plan.all_policies.append(
                PolicyDraft(
                    recommended_actions=[],
                    supi=str(flow_ctx.supi or planning_request.operation_intent.supi or "").strip(),
                    app_id=_normalize_app_id(spec.app_id),
                    flow_id=spec.flow_id,
                    target_type="flow",
                    policy_id=f"smp-{_normalize_app_id(spec.app_id)}-{spec.flow_id}",
                    policy_type="SmPolicyDecision",
                    resource_keys=resource_keys,
                    policy_details=self._build_sm_policy_details(spec, flow_ctx),
                )
            )

        if advisor_output.am_policy is not None:
            plan.all_policies.append(
                PolicyDraft(
                    recommended_actions=[],
                    supi=str(planning_request.operation_intent.supi or "").strip(),
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
                    supi=str((flow_ctx.supi if flow_ctx is not None else "") or planning_request.operation_intent.supi or "").strip(),
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
    def _build_snapshot_writeback_patch(optimizer_preview: Any) -> Dict[str, Any]:
        qos_plan = optimizer_preview.get("qos_plan", {}) if isinstance(optimizer_preview, dict) else {}
        mobility_plan = optimizer_preview.get("mobility_plan", {}) if isinstance(optimizer_preview, dict) else {}
        patch: Dict[str, Any] = {}
        if isinstance(qos_plan, dict):
            patch["qos_plan"] = _json_friendly(
                {
                    "target_app": qos_plan.get("target_app") or {},
                    "target_apps": qos_plan.get("target_apps") or [],
                    "impacted_flows": qos_plan.get("impacted_flows") or [],
                    "slice_stats": qos_plan.get("slice_stats") or [],
                    "meta": qos_plan.get("meta") or {},
                }
            )
        if isinstance(mobility_plan, dict) and mobility_plan:
            patch["mobility_plan"] = _json_friendly(mobility_plan)
        return patch

    @staticmethod
    def _resolve_flow(planning_request: PlanningRequest, flow_id: str) -> FlowSelector:
        target = str(flow_id or "").strip()
        for flow in planning_request.operation_intent.flows:
            if str(flow.flow_id or "").strip() == target:
                return flow
        raise ValueError(f"unknown flow_id in advisor output: {flow_id}")

    @staticmethod
    def _build_sm_policy_details(spec: SmPolicySpec, flow_ctx: FlowSelector) -> Dict[str, Any]:
        flow_id = str(spec.flow_id or "").strip()
        if not flow_id:
            raise ValueError("SmPolicySpec requires flow_id")
        app_id = _normalize_app_id(spec.app_id)
        if not app_id:
            raise ValueError(f"SmPolicySpec requires app_id for flow_id={flow_id}")
        pcc_id = f"pcc-{flow_id}"
        qos_id = f"qos-{flow_id}"
        gbr_ul_mbps = PlanningArtifactCompiler._bounded_gbr_mbps(
            spec.gbr_ul_mbps,
            spec.max_br_ul_mbps,
        )
        gbr_dl_mbps = PlanningArtifactCompiler._bounded_gbr_mbps(
            spec.gbr_dl_mbps,
            spec.max_br_dl_mbps,
        )
        qos_payload: Dict[str, Any] = {
            "qosId": qos_id,
            "5qi": PlanningArtifactCompiler._map_5qi_by_service_type(flow_ctx.service_type_id),
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
        return {
            "_preserve_explicit_qos_values": True,
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
        request_payload["supi"] = str(planning_request.operation_intent.supi or "").strip()
        request_payload["allowedSnssais"] = [item.model_dump(mode="json") for item in spec.allowed_snssais]
        request_payload["targetSnssais"] = [item.model_dump(mode="json") for item in spec.target_snssais]
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
        if spec.serv_area_res is not None:
            request_payload["servAreaRes"] = _json_friendly(spec.serv_area_res)
        elif request_payload.get("servAreaRes") is None and am_policy_context.get("servAreaRes") is not None:
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
        partial_policies: List[Dict[str, Any]],
        planning_request: PlanningRequest,
    ) -> List[PolicyDraft]:
        drafts: List[PolicyDraft] = []
        for index, item in enumerate(partial_policies or [], start=1):
            if not isinstance(item, dict):
                raise TypeError("partial_policies items must be objects")
            policy_type = str(item.get("policy_type") or item.get("policyType") or "").strip()
            if not policy_type:
                raise ValueError("partial_policies items must include policy_type")
            policy_id = str(item.get("policy_id") or item.get("policyId") or "").strip()
            if not policy_id:
                raise ValueError(f"partial_policies[{index}] must include policy_id")
            drafts.append(
                PolicyDraft(
                    recommended_actions=[],
                    supi=str(item.get("supi") or planning_request.operation_intent.supi or "").strip(),
                    app_id=_normalize_app_id(item.get("app_id") or item.get("appId") or ""),
                    flow_id=str(item.get("flow_id") or item.get("flowId") or "").strip() or None,
                    target_type=str(item.get("target_type") or item.get("targetType") or "flow").strip() or "flow",
                    policy_id=policy_id,
                    policy_type=policy_type,
                    resource_keys=[
                        str(value or "").strip()
                        for value in (item.get("resource_keys") or item.get("resourceKeys") or [])
                        if str(value or "").strip()
                    ],
                    policy_details=dict(item.get("policy_details") or item.get("policyDetails") or {}),
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
            supi=str(planning_request.operation_intent.supi or "").strip(),
            session_id=str(planning_request.context.session_id or "").strip(),
            snapshot_id=str(planning_request.context.snapshot_id or "").strip(),
            planning_status="needs_upstream_reground",
            planning_rationale=PlanningRationale(
                selected_strategy_profile=str(
                    planning_request.context.shared_context.initial_intent.objective_profile.get("profile_name")
                    or ""
                ).strip(),
                explanation=rationale,
                unresolved_gaps=[*missing_evidence, *blocked_targets, *upstream_requests],
                main_constraints=[
                    str(item)
                    for item in planning_request.context.shared_context.initial_intent.required_evidence
                    if str(item).strip()
                ],
            ),
            missing_evidence=[str(item) for item in missing_evidence if str(item).strip()],
            blocked_targets=[str(item) for item in blocked_targets if str(item).strip()],
            upstream_requests=[str(item) for item in upstream_requests if str(item).strip()],
            planner_conflicts=[str(item) for item in planner_conflicts if str(item).strip()],
        )


__all__ = ["PlanningArtifactCompiler"]
