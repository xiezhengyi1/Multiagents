from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ...domain.collaboration import PlanningRequest
from ...domain.control_plane import ControlDomain, DomainStatus
from ...domain.policy_plan import FlowSelector, PlanningRationale, PolicyDraft, PolicyPlanDraft, RevisionHandle, RevisionHandles

from .policy_normalizer import json_friendly as _json_friendly
from .policy_normalizer import normalize_app_id as _normalize_app_id
from .policy_normalizer import normalize_policy_plan_draft
from .response_models import AmPolicySpec, OsaAdvisorOutput, SmPolicySpec, UrspPolicySpec


def build_slice_snssai(slice_code: str) -> Optional[Dict[str, Any]]:
    code = str(slice_code or "").strip()
    if len(code) < 8:
        return None
    try:
        sst = int(code[:2], 16)
    except ValueError:
        return None
    return {"sst": sst, "sd": code[2:8]}


class OptimizationStrategyCompiler:
    QOS_GROUNDING_TOOLS = {
        "preview_qos_optimizer",
        "fetch_qos_network_status",
        "preview_optimizer",
        "fetch_network_status",
    }
    MOBILITY_GROUNDING_TOOLS = {
        "inspect_mobility_ue_policies",
        "inspect_ue_policies",
    }

    def build_planning_evidence(self, planning_request: PlanningRequest) -> Dict[str, Any]:
        operation_intent = planning_request.operation_intent
        flows: List[Dict[str, Any]] = []
        for flow in operation_intent.flows:
            flows.append(
                {
                    "flow_id": str(flow.flow_id or "").strip(),
                    "app_id": _normalize_app_id(flow.app_id),
                    "name": str(flow.name or "").strip(),
                    "bw_ul": flow.bw_ul,
                    "bw_dl": flow.bw_dl,
                    "gbr_ul": flow.gbr_ul,
                    "gbr_dl": flow.gbr_dl,
                    "lat": flow.lat,
                    "jitter_req": flow.jitter_req,
                    "loss_req": flow.loss_req,
                    "priority": flow.priority,
                    "service_type_id": flow.service_type_id,
                    "five_tuple": flow.five_tuple,
                }
            )
        qos_objectives = [
            objective.model_dump(mode="json")
            for objective in operation_intent.qos_target_envelopes
        ]
        current_stage = next(
            (
                stage.model_dump(mode="json")
                for stage in (operation_intent.control_semantics.stages or [])
                if int(stage.stage_index or 0) == int(operation_intent.control_semantics.current_stage or 1)
            ),
            {},
        )
        return {
            "requested_domains": list(planning_request.context.active_domains or []),
            "main_retry_scope": str(planning_request.context.main_retry_scope or "").strip(),
            "objective_profile": dict(planning_request.context.objective_profile or {}),
            "required_evidence": list(planning_request.context.required_evidence or []),
            "forbidden_assumptions": list(planning_request.context.forbidden_assumptions or []),
            "revision_requests": list(planning_request.context.revision_requests or []),
            "unified_constraints": dict(planning_request.context.unified_constraints or {}),
            "flows": flows,
            "qos_target_envelopes": qos_objectives,
            "control_semantics": operation_intent.control_semantics.model_dump(mode="json"),
            "current_stage": current_stage,
        }

    def validate_advisor_output(
        self,
        *,
        advisor_output: OsaAdvisorOutput,
        planning_request: PlanningRequest,
        grounding_tools: List[str],
        planning_evidence: Optional[Dict[str, Any]] = None,
        planning_tool_evidence: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        errors: List[str] = []
        domains = {
            str(item).strip().lower()
            for item in (planning_request.context.active_domains or [])
            if str(item).strip()
        }
        grounding = set(grounding_tools or [])
        required_evidence = {
            str(item).strip().lower()
            for item in (planning_request.context.required_evidence or [])
            if str(item).strip()
        }
        normalized_tool_evidence = dict(planning_tool_evidence or {})
        retry_scope = str(planning_request.context.main_retry_scope or "").strip().lower()
        preserved_app_id = self._preserved_app_id(planning_request)
        preserved_flow_ids = self._preserved_flow_ids(planning_request)

        has_sm = bool(advisor_output.sm_policies)
        has_am = advisor_output.am_policy is not None
        has_ursp = bool(advisor_output.ursp_policies)

        if has_sm and "qos" not in domains:
            errors.append("advisor emitted sm_policies outside qos-active planning")
        if has_am and "mobility" not in domains:
            errors.append("advisor emitted am_policy outside mobility-active planning")
        if "qos" in domains and not has_sm:
            errors.append("qos-active planning requires sm_policies")
        if "mobility" in domains and not has_am:
            errors.append("mobility-active planning requires am_policy")
        if required_evidence and not grounding:
            errors.append("planning context requires evidence collection but advisor used no grounding tools")
        optimizer_preview = self._latest_optimizer_preview(normalized_tool_evidence)
        mobility_context = self._latest_mobility_context(normalized_tool_evidence)
        has_ursp_signal = self._contains_ursp_signal(
            planning_request.operation_intent.raw_input,
            planning_request.operation_intent.mobility_intent,
            planning_request.operation_intent.domain_evidence,
            planning_request.context.revision_requests,
        )
        has_existing_policy_evidence = (
            (not has_sm or bool(optimizer_preview))
            and (not has_am or bool(mobility_context))
            and (not has_ursp or has_ursp_signal)
        )
        if (has_sm or has_am or has_ursp) and not grounding and not has_existing_policy_evidence:
            errors.append("policy output requires at least one non-think grounding tool")
        if has_sm and "preview_qos_optimizer" not in grounding:
            errors.append("sm_policies require preview_qos_optimizer evidence in the same round")
        if has_sm and not optimizer_preview:
            errors.append("sm_policies require a parseable optimizer preview payload")
        elif has_sm:
            preview_errors = self._validate_optimizer_preview_payload(optimizer_preview)
            errors.extend(preview_errors)
            for spec in advisor_output.sm_policies:
                if not self._extract_qos_resource_keys(optimizer_preview, flow_id=spec.flow_id):
                    errors.append(
                        f"optimizer preview does not contain a grounded QoS assignment for flow_id={spec.flow_id}"
                    )
        if has_am and not (self.MOBILITY_GROUNDING_TOOLS & grounding):
            errors.append("am_policy requires inspect_mobility_ue_policies evidence")
        if has_am and not mobility_context:
            errors.append("am_policy requires a parseable mobility context payload")
        if has_ursp:
            if not has_ursp_signal:
                errors.append("ursp_policies require explicit route-selection or UE-policy-routing intent")
            if not has_ursp_signal and not (
                (self.MOBILITY_GROUNDING_TOOLS & grounding)
                or ({"search_semantic_knowledge", "get_knowledge_by_key"} & grounding)
            ):
                errors.append("ursp_policies require explicit routing or policy-semantic evidence")
        if retry_scope == "policy_repair":
            if preserved_app_id:
                for index, spec in enumerate(advisor_output.sm_policies):
                    if _normalize_app_id(spec.app_id) != preserved_app_id:
                        errors.append(
                            f"policy_repair sm_policies[{index}] must preserve app_id={preserved_app_id}; got {spec.app_id}"
                        )
            if preserved_flow_ids:
                advisor_flow_ids = {str(item.flow_id or "").strip() for item in advisor_output.sm_policies if str(item.flow_id or "").strip()}
                if advisor_flow_ids and advisor_flow_ids != preserved_flow_ids:
                    errors.append(
                        "policy_repair sm_policies must preserve grounded flow_ids "
                        f"{sorted(preserved_flow_ids)}; got {sorted(advisor_flow_ids)}"
                    )
        return errors

    @staticmethod
    def _contains_ursp_signal(*parts: Any) -> bool:
        raw_text = " ".join(
            [
                json.dumps(_json_friendly(part), ensure_ascii=False) if not isinstance(part, str) else str(part or "")
                for part in parts
            ]
        ).lower()
        return any(token in raw_text for token in ("ursp", "route", "routing", "route selection", "ue policy"))

    def assemble_policy_plan(
        self,
        *,
        advisor_output: OsaAdvisorOutput,
        planning_request: PlanningRequest,
        planning_tool_evidence: Dict[str, Any],
    ) -> PolicyPlanDraft:
        optimizer_preview = self._latest_optimizer_preview(planning_tool_evidence)
        mobility_context = self._latest_mobility_context(planning_tool_evidence)
        if advisor_output.sm_policies:
            self._validate_optimizer_preview(optimizer_preview)
        if advisor_output.am_policy is not None and not mobility_context:
            raise ValueError("am_policy compilation requires grounded mobility context")

        planning_metadata = {
            "planning_mode": "advisor_compiler",
            "requested_domains": list(planning_request.context.active_domains or []),
            "main_retry_scope": str(planning_request.context.main_retry_scope or "").strip(),
            "objective_breakdown": dict(optimizer_preview.get("objective_breakdown") or {}) if isinstance(optimizer_preview, dict) else {},
            "advisor_rationale": advisor_output.rationale,
            "advisor_metadata": _json_friendly(advisor_output.planning_metadata),
            "revision_requests": _json_friendly(planning_request.context.revision_requests or []),
            "unified_constraints": _json_friendly(planning_request.context.unified_constraints or {}),
            "optimizer_cross_domain_verdicts": [
                _json_friendly(item)
                for item in ((optimizer_preview.get("cross_domain_verdicts") if isinstance(optimizer_preview, dict) else []) or [])
            ],
            "snapshot_writeback_patch": self._build_snapshot_writeback_patch(optimizer_preview),
        }

        plan = PolicyPlanDraft(
            supi=str(planning_request.operation_intent.supi or "").strip(),
            session_id=str(planning_request.context.session_id or "").strip(),
            snapshot_id=str(planning_request.context.snapshot_id or "").strip(),
            planning_metadata=planning_metadata,
            planning_rationale=PlanningRationale(
                selected_strategy_profile=str(
                    planning_request.context.objective_profile.get("profile_name")
                    or planning_request.operation_intent.objective_profile_hint
                    or ""
                ).strip(),
                objective_tradeoff_summary=str(
                    advisor_output.rationale
                    or planning_metadata["objective_breakdown"]
                    or ""
                ).strip(),
                decisive_evidence=[
                    item
                    for item in [
                        "main_retry_scope:policy_repair" if str(planning_request.context.main_retry_scope or "").strip().lower() == "policy_repair" else "",
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
                        "identifier_rebinding" if str(planning_request.context.main_retry_scope or "").strip().lower() == "policy_repair" else "",
                    ]
                    if item
                ],
            ),
            revision_handles=RevisionHandles(handles=[]),
            all_policies=[],
        )

        for spec in advisor_output.sm_policies:
            flow_ctx = self._resolve_flow(planning_request, spec.flow_id)
            if _normalize_app_id(spec.app_id) != _normalize_app_id(flow_ctx.app_id or planning_request.operation_intent.app_id or ""):
                raise ValueError(f"sm policy app_id does not match resolved flow context for flow_id={spec.flow_id}")
            resource_keys = self._extract_qos_resource_keys(optimizer_preview, flow_id=spec.flow_id)
            if not resource_keys:
                raise ValueError(f"optimizer preview does not contain a grounded QoS assignment for flow_id={spec.flow_id}")
            plan.all_policies.append(
                PolicyDraft(
                    recommended_actions=[spec.flow_description or f"Apply QoS strategy for {spec.flow_id}"],
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
                    recommended_actions=[advisor_output.am_policy.rationale] if advisor_output.am_policy.rationale else [],
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
                    recommended_actions=[spec.rationale] if spec.rationale else [],
                    supi=str((flow_ctx.supi if flow_ctx is not None else "") or planning_request.operation_intent.supi or "").strip(),
                    app_id=_normalize_app_id(spec.app_id),
                    flow_id=spec.flow_id,
                    target_type=spec.target_type,
                    policy_id=f"ursp-{_normalize_app_id(spec.app_id)}-{spec.flow_id or index}",
                    policy_type="UrspRuleRequest",
                    policy_details=self._build_ursp_policy_details(spec),
                )
            )

        normalized = normalize_policy_plan_draft(plan, planning_request.operation_intent)
        normalized.revision_handles = self._build_revision_handles(
            policy_plan=normalized,
            planning_request=planning_request,
        )
        self._validate_compiled_plan(normalized, planning_request)
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
    def _validate_optimizer_preview(optimizer_preview: Any) -> None:
        if not isinstance(optimizer_preview, dict) or not optimizer_preview:
            raise ValueError("missing grounded optimizer preview")
        result_status = str(optimizer_preview.get("status") or "").strip().lower()
        infeasible_reasons = list(optimizer_preview.get("infeasible_reasons", []) or [])
        qos_plan = optimizer_preview.get("qos_plan") if isinstance(optimizer_preview, dict) else {}
        qos_meta = qos_plan.get("meta") if isinstance(qos_plan, dict) else {}
        qos_meta_status = str((qos_meta or {}).get("status") or "").strip()
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
        qos_payload: Dict[str, Any] = {
            "qosId": qos_id,
            "priorityLevel": int(spec.priority),
            "packetDelayBudget": int(spec.target_latency_ms),
            "packetErrorRate": str(spec.packet_error_rate),
            "maxbrUl": str(spec.max_br_ul_mbps),
            "maxbrDl": str(spec.max_br_dl_mbps),
        }
        if spec.gbr_ul_mbps is not None:
            qos_payload["gbrUl"] = str(spec.gbr_ul_mbps)
        if spec.gbr_dl_mbps is not None:
            qos_payload["gbrDl"] = str(spec.gbr_dl_mbps)
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

    def _resolve_am_association_id(self, *, planning_request: PlanningRequest, mobility_context: Dict[str, Any]) -> str:
        preserved = self._preserved_association_id(planning_request)
        if preserved:
            return preserved
        mobility_summary = mobility_context.get("mobilitySummary") if isinstance(mobility_context, dict) else {}
        association_id = str((mobility_summary or {}).get("currentAssociationId") or "").strip()
        if not association_id:
            am_policy_context = mobility_context.get("amPolicyContext") if isinstance(mobility_context, dict) else {}
            associations = (am_policy_context or {}).get("associations")
            if isinstance(associations, dict) and len(associations) == 1:
                association_id = str(next(iter(associations.keys())) or "").strip()
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

    def _validate_compiled_plan(self, policy_plan: PolicyPlanDraft, planning_request: PlanningRequest) -> None:
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
        if retry_scope == "policy_repair":
            preserved_app_id = self._preserved_app_id(planning_request)
            preserved_flow_ids = self._preserved_flow_ids(planning_request)
            preserved_association_id = self._preserved_association_id(planning_request)
            if preserved_app_id:
                drifted_policy_ids = [
                    item.policy_id
                    for item in policy_plan.all_policies
                    if item.policy_type == "SmPolicyDecision" and _normalize_app_id(item.app_id) != preserved_app_id
                ]
                if drifted_policy_ids:
                    raise ValueError(
                        f"policy_repair compiled SM policies changed app_id away from preserved {preserved_app_id}: {drifted_policy_ids}"
                    )
            if preserved_flow_ids:
                compiled_flow_ids = {
                    str(item.flow_id or "").strip()
                    for item in policy_plan.all_policies
                    if item.policy_type == "SmPolicyDecision" and str(item.flow_id or "").strip()
                }
                if compiled_flow_ids != preserved_flow_ids:
                    raise ValueError(
                        "policy_repair compiled SM policies changed flow_ids away from preserved "
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
                        "policy_repair compiled AM policy changed association_id away from preserved "
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
        return list(dict.fromkeys(keys))

    @staticmethod
    def _latest_optimizer_preview(planning_tool_evidence: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(planning_tool_evidence.get("latest_optimizer_preview") or {})
        result = payload.get("result")
        if isinstance(result, dict):
            return dict(result)
        if isinstance(payload, dict) and payload.get("status") is not None:
            return payload
        return {}

    @staticmethod
    def _latest_mobility_context(planning_tool_evidence: Dict[str, Any]) -> Dict[str, Any]:
        return dict(planning_tool_evidence.get("latest_mobility_context") or {})

    @staticmethod
    def _build_revision_handles(
        *,
        policy_plan: PolicyPlanDraft,
        planning_request: PlanningRequest,
    ) -> RevisionHandles:
        qos_policy_ids = [item.policy_id for item in policy_plan.all_policies if item.policy_type == "SmPolicyDecision"]
        qos_flow_ids = [str(item.flow_id or "").strip() for item in policy_plan.all_policies if item.policy_type == "SmPolicyDecision" and str(item.flow_id or "").strip()]
        mobility_policy_ids = [item.policy_id for item in policy_plan.all_policies if item.policy_type == "PcfAmPolicyControlPolicyAssociation"]
        handles: List[RevisionHandle] = []
        if qos_policy_ids:
            handles.append(
                RevisionHandle(
                    scope="qos",
                    target_policy_ids=qos_policy_ids,
                    target_flow_ids=qos_flow_ids,
                    required_recompute=["optimizer_preview_qos"],
                    rationale="QoS policies can be revised without rebuilding mobility payloads when identifiers stay preserved.",
                )
            )
        if mobility_policy_ids:
            handles.append(
                RevisionHandle(
                    scope="mobility",
                    target_policy_ids=mobility_policy_ids,
                    target_flow_ids=[],
                    required_recompute=["optimizer_preview_mobility"],
                    rationale="Mobility policy fields can be revised from preserved AM context and optimizer mobility evidence.",
                )
            )
        if len(handles) > 1 or planning_request.context.unified_constraints:
            handles.append(
                RevisionHandle(
                    scope="joint_coupling",
                    target_policy_ids=[item.policy_id for item in policy_plan.all_policies],
                    target_flow_ids=qos_flow_ids,
                    required_recompute=["optimizer_preview_joint", "mediator_constraints"],
                    rationale="Cross-domain constraints or mixed-domain plans require joint coupling-aware revision.",
                )
            )
        return RevisionHandles(handles=handles)

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
    def _preserved_association_id(planning_request: PlanningRequest) -> str:
        mobility_targets = planning_request.operation_intent.grounding_evidence.grounded_mobility_targets
        if isinstance(mobility_targets, dict):
            summary = mobility_targets.get("summary")
            if isinstance(summary, dict):
                return str(summary.get("current_association_id") or "").strip()
        return ""


__all__ = ["OptimizationStrategyCompiler", "build_slice_snssai"]
