from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from domain.collaboration import PlanningRequest
from domain.control_plane import ControlDomain, DomainStatus
from domain.policy_plan import FlowSelector, PolicyDraft, PolicyPlanDraft

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
    def build_planning_evidence(self, planning_request: PlanningRequest, optimizer_preview: Any) -> Dict[str, Any]:
        operation_intent = planning_request.operation_intent
        preview_payload = (
            optimizer_preview.model_dump(mode="json")
            if hasattr(optimizer_preview, "model_dump")
            else _json_friendly(optimizer_preview)
        )
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
        return {
            "requested_domains": list(planning_request.context.active_domains or []),
            "objective_profile": dict(planning_request.context.objective_profile or {}),
            "required_evidence": list(planning_request.context.required_evidence or []),
            "forbidden_assumptions": list(planning_request.context.forbidden_assumptions or []),
            "revision_requests": list(planning_request.context.revision_requests or []),
            "unified_constraints": dict(planning_request.context.unified_constraints or {}),
            "flows": flows,
            "preview_status": preview_payload.get("status") if isinstance(preview_payload, dict) else None,
            "preview_objective_breakdown": preview_payload.get("objective_breakdown") if isinstance(preview_payload, dict) else {},
            "preview_infeasible_reasons": preview_payload.get("infeasible_reasons") if isinstance(preview_payload, dict) else [],
            "preview_qos_plan_present": bool(preview_payload.get("qos_plan")) if isinstance(preview_payload, dict) else False,
            "preview_mobility_plan_present": bool(preview_payload.get("mobility_plan")) if isinstance(preview_payload, dict) else False,
        }

    def validate_advisor_output(
        self,
        *,
        advisor_output: OsaAdvisorOutput,
        planning_request: PlanningRequest,
        grounding_tools: List[str],
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
        if (has_sm or has_am or has_ursp) and not grounding:
            errors.append("policy output requires at least one non-think grounding tool")
        if has_sm and not ({"preview_optimizer", "fetch_network_status"} & grounding):
            errors.append("sm_policies require preview_optimizer or fetch_network_status evidence")
        if has_am and "inspect_ue_policies" not in grounding:
            errors.append("am_policy requires inspect_ue_policies evidence")
        if has_ursp:
            if not self._ursp_requested_or_evidenced(planning_request):
                errors.append("ursp_policies require explicit route-selection or UE-policy-routing intent")
            if not ({"inspect_ue_policies", "search_semantic_knowledge", "get_knowledge_by_key"} & grounding):
                errors.append("ursp_policies require explicit routing or policy-semantic evidence")
        return errors

    @staticmethod
    def _ursp_requested_or_evidenced(planning_request: PlanningRequest) -> bool:
        raw_text = " ".join(
            [
                str(planning_request.operation_intent.raw_input or ""),
                json.dumps(_json_friendly(planning_request.operation_intent.mobility_intent or {}), ensure_ascii=False),
                json.dumps(_json_friendly(planning_request.operation_intent.domain_evidence or {}), ensure_ascii=False),
                json.dumps(_json_friendly(planning_request.context.revision_requests or []), ensure_ascii=False),
            ]
        ).lower()
        return any(token in raw_text for token in ("ursp", "route", "routing", "route selection", "ue policy"))

    def assemble_policy_plan(
        self,
        *,
        advisor_output: OsaAdvisorOutput,
        planning_request: PlanningRequest,
        optimizer_preview: Any,
    ) -> PolicyPlanDraft:
        self._validate_optimizer_preview(optimizer_preview)

        planning_metadata = {
            "planning_mode": "advisor_compiler",
            "requested_domains": list(planning_request.context.active_domains or []),
            "objective_breakdown": getattr(optimizer_preview, "objective_breakdown", {}) or {},
            "advisor_rationale": advisor_output.rationale,
            "advisor_metadata": _json_friendly(advisor_output.planning_metadata),
            "revision_requests": _json_friendly(planning_request.context.revision_requests or []),
            "unified_constraints": _json_friendly(planning_request.context.unified_constraints or {}),
            "optimizer_cross_domain_verdicts": [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else _json_friendly(item)
                for item in (getattr(optimizer_preview, "cross_domain_verdicts", []) or [])
            ],
        }

        plan = PolicyPlanDraft(
            supi=str(planning_request.operation_intent.supi or "").strip(),
            session_id=str(planning_request.context.session_id or "").strip(),
            snapshot_id=str(planning_request.context.snapshot_id or "").strip(),
            planning_metadata=planning_metadata,
            all_policies=[],
        )

        for spec in advisor_output.sm_policies:
            flow_ctx = self._resolve_flow(planning_request, spec.flow_id)
            if _normalize_app_id(spec.app_id) != _normalize_app_id(flow_ctx.app_id or planning_request.operation_intent.app_id or ""):
                raise ValueError(f"sm policy app_id does not match resolved flow context for flow_id={spec.flow_id}")
            plan.all_policies.append(
                PolicyDraft(
                    recommended_actions=[spec.flow_description or f"Apply QoS strategy for {spec.flow_id}"],
                    supi=str(planning_request.operation_intent.supi or "").strip(),
                    app_id=_normalize_app_id(spec.app_id),
                    flow_id=spec.flow_id,
                    target_type="flow",
                    policy_id=f"smp-{_normalize_app_id(spec.app_id)}-{spec.flow_id}",
                    policy_type="SmPolicyDecision",
                    resource_keys=self._extract_qos_resource_keys(optimizer_preview, flow_id=spec.flow_id),
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
                    policy_id=self._resolve_am_association_id(optimizer_preview),
                    policy_type="PcfAmPolicyControlPolicyAssociation",
                    policy_details=self._build_am_policy_details(
                        advisor_output.am_policy,
                        planning_request=planning_request,
                        optimizer_preview=optimizer_preview,
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
                    supi=str(planning_request.operation_intent.supi or "").strip(),
                    app_id=_normalize_app_id(spec.app_id),
                    flow_id=spec.flow_id,
                    target_type=spec.target_type,
                    policy_id=f"ursp-{_normalize_app_id(spec.app_id)}-{spec.flow_id or index}",
                    policy_type="UrspRuleRequest",
                    policy_details=self._build_ursp_policy_details(spec),
                )
            )

        normalized = normalize_policy_plan_draft(plan, planning_request.operation_intent)
        self._validate_compiled_plan(normalized, planning_request)
        return normalized

    @staticmethod
    def _validate_optimizer_preview(optimizer_preview: Any) -> None:
        result_status = getattr(optimizer_preview, "status", None)
        infeasible_reasons = list(getattr(optimizer_preview, "infeasible_reasons", []) or [])
        if result_status == DomainStatus.INCOMPLETE_CONTEXT:
            reason_text = "; ".join(str(item) for item in infeasible_reasons) or "missing required planning context"
            raise ValueError(f"incomplete_context: {reason_text}")
        if infeasible_reasons:
            raise ValueError("Joint optimizer returned infeasible result: " + "; ".join(str(item) for item in infeasible_reasons))

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

    @staticmethod
    def _resolve_am_association_id(optimizer_preview: Any) -> str:
        mobility_plan = getattr(optimizer_preview, "mobility_plan", {}) or {}
        association_id = str(mobility_plan.get("association_id") or "").strip()
        if not association_id:
            raise ValueError("optimizer preview does not contain mobility association_id")
        return association_id

    def _build_am_policy_details(
        self,
        spec: AmPolicySpec,
        *,
        planning_request: PlanningRequest,
        optimizer_preview: Any,
    ) -> Dict[str, Any]:
        mobility_plan = getattr(optimizer_preview, "mobility_plan", {}) or {}
        request_payload = mobility_plan.get("request")
        policy_payload = mobility_plan.get("policy")
        if not isinstance(request_payload, dict) or not request_payload:
            raise ValueError("optimizer preview does not contain a grounded AM request payload")
        if not isinstance(policy_payload, dict) or not policy_payload:
            raise ValueError("optimizer preview does not contain a grounded AM policy payload")
        if not str(request_payload.get("notificationUri") or "").strip():
            raise ValueError("grounded AM request payload is missing notificationUri")

        request_payload = _json_friendly(request_payload)
        policy_payload = _json_friendly(policy_payload)
        request_payload["supi"] = str(planning_request.operation_intent.supi or "").strip()

        # 关键步骤：优先使用 MILP am_plan 中的 AM 决策参数，否则回退到 LLM 生成的 spec
        am_plan = getattr(optimizer_preview, "am_plan", {}) or {}
        if am_plan.get("allowed_snssais"):
            request_payload["allowedSnssais"] = [
                build_slice_snssai(s) for s in am_plan["allowed_snssais"]
                if build_slice_snssai(s) is not None
            ]
        else:
            request_payload["allowedSnssais"] = [item.model_dump(mode="json") for item in spec.allowed_snssais]

        if am_plan.get("target_snssais"):
            request_payload["targetSnssais"] = [
                build_slice_snssai(s) for s in am_plan["target_snssais"]
                if build_slice_snssai(s) is not None
            ]
        else:
            request_payload["targetSnssais"] = [item.model_dump(mode="json") for item in spec.target_snssais]

        request_payload["rfsp"] = am_plan.get("rfsp") or int(spec.rfsp)

        if am_plan.get("ue_ambr_ul_mbps") is not None and am_plan.get("ue_ambr_dl_mbps") is not None:
            request_payload["ueAmbr"] = {
                "uplink": str(am_plan["ue_ambr_ul_mbps"]),
                "downlink": str(am_plan["ue_ambr_dl_mbps"]),
            }
        elif spec.ue_ambr_ul_mbps is not None or spec.ue_ambr_dl_mbps is not None:
            if spec.ue_ambr_ul_mbps is None or spec.ue_ambr_dl_mbps is None:
                raise ValueError("ue_ambr_ul_mbps and ue_ambr_dl_mbps must be provided together")
            request_payload["ueAmbr"] = {
                "uplink": str(spec.ue_ambr_ul_mbps),
                "downlink": str(spec.ue_ambr_dl_mbps),
            }
        if spec.serv_area_res is not None:
            request_payload["servAreaRes"] = _json_friendly(spec.serv_area_res)

        policy_payload["request"] = request_payload
        policy_payload["triggers"] = am_plan.get("triggers") or list(spec.triggers)
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
        if ControlDomain.QOS.value in active_domains and ControlDomain.MOBILITY.value in active_domains:
            self._validate_joint_snssai_consistency(policy_plan)

    @staticmethod
    def _validate_joint_snssai_consistency(policy_plan: PolicyPlanDraft) -> None:
        selected_snssais: set[str] = set()
        allowed_snssais: set[str] = set()
        for item in policy_plan.all_policies:
            if item.policy_type == "SmPolicyDecision":
                for resource_key in item.resource_keys or []:
                    if str(resource_key).startswith("snssai:"):
                        selected_snssais.add(str(resource_key))
            elif item.policy_type == "PcfAmPolicyControlPolicyAssociation":
                request = item.policy_details.get("request") if isinstance(item.policy_details, dict) else {}
                for snssai in request.get("allowedSnssais") or []:
                    allowed_snssais.add(f"snssai:{json.dumps(snssai, sort_keys=True, ensure_ascii=False)}")
        uncovered = sorted(selected_snssais - allowed_snssais)
        if uncovered:
            raise ValueError(
                "QoS-selected S-NSSAI values are not covered by mobility allowedSnssais: "
                + ", ".join(uncovered)
            )

    def _extract_qos_resource_keys(self, joint_result: Any, *, flow_id: str) -> List[str]:
        qos_plan = getattr(joint_result, "qos_plan", {}) or {}
        target_app = qos_plan.get("target_app") if isinstance(qos_plan, dict) else {}
        flows = target_app.get("flows") if isinstance(target_app, dict) else []
        if not isinstance(flows, list):
            return []

        keys: List[str] = []
        for item in flows:
            if not isinstance(item, dict):
                continue
            if str(item.get("flow_id") or "").strip() != flow_id:
                continue
            selected_slice = str(item.get("New Slice") or "").strip()
            if selected_slice:
                keys.append(f"slice:{selected_slice}")
                snssai = build_slice_snssai(selected_slice)
                if snssai is not None:
                    keys.append(f"snssai:{json.dumps(snssai, sort_keys=True, ensure_ascii=False)}")
        return list(dict.fromkeys(keys))


__all__ = ["OptimizationStrategyCompiler", "build_slice_snssai"]
