from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from domain.control_plane import (
    ControlDomain,
    DomainStatus,
    DomainVerdict,
    JointOptimizationRequest,
    MobilityContextSnapshot,
    MobilityPolicyDraft,
    OptimizationProblemConfig,
)
from model.PcfAmPolicyControl import (
    AccessType,
    Ambr,
    MappingOfSnssai,
    PcfAmPolicyControlPolicyAssociation,
    PcfAmPolicyControlPolicyAssociationRequest,
    PcfAmPolicyControlRequestTrigger,
    PresenceInfo,
    RatType,
    ServiceAreaRestriction,
    SmfSelectionData,
    Snssai,
    UserLocation,
)


REQUIRED_MOBILITY_FIELDS = ("accessType", "userLoc", "guami", "servingPlmn")


def coerce_snssai_list(raw_items: Any) -> List[Snssai]:
    result: List[Snssai] = []
    if not isinstance(raw_items, list):
        return result
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            result.append(Snssai.model_validate(item))
        except Exception:
            continue
    return result


def coerce_mapping_snssai_list(raw_items: Any) -> List[MappingOfSnssai]:
    result: List[MappingOfSnssai] = []
    if not isinstance(raw_items, list):
        return result
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            result.append(MappingOfSnssai.model_validate(item))
        except Exception:
            continue
    return result


def coerce_presence_map(raw_value: Any) -> Dict[str, PresenceInfo]:
    result: Dict[str, PresenceInfo] = {}
    if not isinstance(raw_value, dict):
        return result
    for key, value in raw_value.items():
        if not isinstance(value, dict):
            continue
        try:
            result[str(key)] = PresenceInfo.model_validate(value)
        except Exception:
            continue
    return result


def build_mobility_snapshot(request: JointOptimizationRequest, supi: str) -> MobilityContextSnapshot:
    mobility_state = request.mobility_state or {}
    raw_ctx: Dict[str, Any] = {}

    if isinstance(mobility_state, dict):
        if isinstance(mobility_state.get(supi), dict):
            raw_ctx = dict(mobility_state.get(supi) or {})
        elif isinstance(mobility_state.get("ues"), list):
            for item in mobility_state["ues"]:
                if isinstance(item, dict) and str(item.get("supi") or "").strip() == supi:
                    raw_ctx = dict(item)
                    break

    missing_fields = [name for name in REQUIRED_MOBILITY_FIELDS if not raw_ctx.get(name)]

    return MobilityContextSnapshot(
        supi=supi,
        accessType=AccessType(raw_ctx["accessType"]) if raw_ctx.get("accessType") else None,
        accessTypes=[AccessType(item) for item in raw_ctx.get("accessTypes", []) if item],
        ratType=RatType(raw_ctx["ratType"]) if raw_ctx.get("ratType") else None,
        ratTypes=[RatType(item) for item in raw_ctx.get("ratTypes", []) if item],
        userLoc=UserLocation.model_validate(raw_ctx["userLoc"]) if isinstance(raw_ctx.get("userLoc"), dict) else None,
        guami=raw_ctx.get("guami"),
        servingPlmn=raw_ctx.get("servingPlmn"),
        timeZone=raw_ctx.get("timeZone"),
        presenceAreas=coerce_presence_map(raw_ctx.get("presenceAreas")),
        allowedSnssais=coerce_snssai_list(raw_ctx.get("allowedSnssais")),
        targetSnssais=coerce_snssai_list(raw_ctx.get("targetSnssais")),
        mappingSnssais=coerce_mapping_snssai_list(raw_ctx.get("mappingSnssais")),
        currentAssociationId=raw_ctx.get("currentAssociationId"),
        currentTriggers=[PcfAmPolicyControlRequestTrigger(item) for item in raw_ctx.get("currentTriggers", []) if item],
        missing_fields=missing_fields,
    )


def build_slice_snssai(slice_code: str) -> Optional[Snssai]:
    code = str(slice_code or "").strip()
    if len(code) < 8:
        return None
    try:
        sst = int(code[:2], 16)
    except ValueError:
        return None
    sd = code[2:8]
    return Snssai(sst=sst, sd=sd)


def select_target_snssais(snapshot: MobilityContextSnapshot, qos_plan: Dict[str, Any]) -> List[Snssai]:
    target: List[Snssai] = list(snapshot.targetSnssais)
    if target:
        return target
    for flow in qos_plan.get("target_app", {}).get("flows", []):
        if not isinstance(flow, dict):
            continue
        candidate = build_slice_snssai((flow.get("allocation") or {}).get("current_slice_snssai"))
        if candidate is not None and all(item.model_dump() != candidate.model_dump() for item in target):
            target.append(candidate)
    if not target:
        target = list(snapshot.allowedSnssais)
    return target


def build_mobility_draft(
    request: JointOptimizationRequest,
    supi: str,
    snapshot: MobilityContextSnapshot,
    qos_plan: Dict[str, Any],
    *,
    am_plan: Optional[Dict[str, Any]] = None,
) -> MobilityPolicyDraft:
    association_id = snapshot.currentAssociationId or f"{supi}-am-assoc-1"

    # 关键步骤：优先使用 MILP 求解的 AM 最优解
    if am_plan and am_plan.get("allowed_snssais"):
        allowed_snssais = [s for s in (build_slice_snssai(code) for code in am_plan["allowed_snssais"]) if s is not None]
        target_snssais = [s for s in (build_slice_snssai(code) for code in am_plan.get("target_snssais", [])) if s is not None]
        if not target_snssais:
            target_snssais = allowed_snssais
    else:
        target_snssais = select_target_snssais(snapshot, qos_plan)
        allowed_snssais = list(snapshot.allowedSnssais or target_snssais)
    if not allowed_snssais:
        allowed_snssais = target_snssais

    required_triggers = [
        PcfAmPolicyControlRequestTrigger.LOC_CH,
        PcfAmPolicyControlRequestTrigger.PRA_CH,
        PcfAmPolicyControlRequestTrigger.ALLOWED_NSSAI_CH,
    ]
    # 关键步骤：若 MILP 解包含 triggers，用 MILP 结果覆盖
    if am_plan and am_plan.get("triggers"):
        required_triggers = []
        for trig_name in am_plan["triggers"]:
            try:
                required_triggers.append(PcfAmPolicyControlRequestTrigger(trig_name))
            except Exception:
                continue
    else:
        for trigger in request.operation_intent.get("mobility_triggers", []) if isinstance(request.operation_intent, dict) else []:
            try:
                enum_trigger = PcfAmPolicyControlRequestTrigger(trigger)
            except Exception:
                continue
            if enum_trigger not in required_triggers:
                required_triggers.append(enum_trigger)
    if not snapshot.presenceAreas:
        required_triggers = [
            trigger
            for trigger in required_triggers
            if trigger != PcfAmPolicyControlRequestTrigger.PRA_CH
        ]

    # 关键步骤：若 MILP 解包含 AMBR，则使用 MILP 值
    if am_plan and am_plan.get("ue_ambr_ul_mbps") is not None and am_plan.get("ue_ambr_dl_mbps") is not None:
        total_ul = float(am_plan["ue_ambr_ul_mbps"])
        total_dl = float(am_plan["ue_ambr_dl_mbps"])
    else:
        total_ul = 0.0
        total_dl = 0.0
        for flow in request.operation_intent.get("flows", []) if isinstance(request.operation_intent, dict) else []:
            if not isinstance(flow, dict):
                continue
            total_ul += float(flow.get("bw_ul") or 0.0)
            total_dl += float(flow.get("bw_dl") or 0.0)

    rfsp_value = (am_plan.get("rfsp") if am_plan else None) or max(1, (snapshot.currentRfsp or 1))

    ue_ambr = Ambr(uplink=f"{max(total_ul, 1.0):.1f} Mbps", downlink=f"{max(total_dl, 1.0):.1f} Mbps")
    ue_slice_mbrs = [
        {
            "sliceMbr": {"default": {"uplink": f"{max(total_ul, 1.0):.1f} Mbps", "downlink": f"{max(total_dl, 1.0):.1f} Mbps"}},
            "servingSnssai": item.model_dump(mode="json"),
        }
        for item in target_snssais
    ]

    request_payload = PcfAmPolicyControlPolicyAssociationRequest(
        notificationUri=f"http://localhost:18080/notify/{supi}",
        supi=supi,
        accessType=snapshot.accessType,
        accessTypes=snapshot.accessTypes or ([snapshot.accessType] if snapshot.accessType else None),
        userLoc=snapshot.userLoc,
        timeZone=snapshot.timeZone or "+08:00",
        servingPlmn=snapshot.servingPlmn,
        ratType=snapshot.ratType,
        ratTypes=snapshot.ratTypes or ([snapshot.ratType] if snapshot.ratType else None),
        servAreaRes=ServiceAreaRestriction(restrictionType="ALLOWED_AREAS", areas=[]),
        rfsp=rfsp_value,
        ueAmbr=ue_ambr,
        allowedSnssais=allowed_snssais,
        targetSnssais=target_snssais,
        mappingSnssais=snapshot.mappingSnssais or [
            MappingOfSnssai(servingSnssai=item, homeSnssai=item) for item in target_snssais
        ],
        guami=snapshot.guami,
        suppFeat="1",
    )
    policy_payload = PcfAmPolicyControlPolicyAssociation(
        request=request_payload,
        triggers=required_triggers,
        servAreaRes=request_payload.servAreaRes,
        rfsp=request_payload.rfsp,
        smfSelInfo=SmfSelectionData(snssai=target_snssais[0] if target_snssais else None),
        ueAmbr=ue_ambr,
        ueSliceMbrs=ue_slice_mbrs,
        pras=snapshot.presenceAreas or {},
        suppFeat="1",
    )
    return MobilityPolicyDraft(
        association_id=association_id,
        request=request_payload,
        policy=policy_payload,
        rationale="Derived AM policy draft from current UE access state, allowed/target NSSAI, and requested QoS outcome.",
        trigger_event="JOINT_CONTROL_REEVALUATION",
        expected_benefits=[
            "keep AM policy aligned with requested slice selection",
            "reduce mobility-policy mismatch risk during subsequent access changes",
        ],
    )


def snssai_key(item: Snssai) -> Tuple[int, Optional[str]]:
    return item.sst, item.sd


def run_cross_domain_checks(
    snapshot: MobilityContextSnapshot,
    qos_plan: Dict[str, Any],
    mobility_draft: Optional[MobilityPolicyDraft],
    *,
    problem_config: OptimizationProblemConfig,
) -> List[DomainVerdict]:
    verdicts: List[DomainVerdict] = []
    if mobility_draft is None:
        return verdicts

    hard_conflicts: List[str] = []
    soft_conflicts: List[str] = []

    target_snssais = mobility_draft.request.targetSnssais or []
    allowed_snssais = mobility_draft.request.allowedSnssais or []
    allowed_keys = {snssai_key(item) for item in allowed_snssais}
    if "snssai_alignment" in problem_config.active_constraints:
        for item in target_snssais:
            if snssai_key(item) not in allowed_keys:
                hard_conflicts.append("targetSnssais must be a subset of allowedSnssais")

        for flow in qos_plan.get("target_app", {}).get("flows", []):
            if not isinstance(flow, dict):
                continue
            snssai = build_slice_snssai((flow.get("allocation") or {}).get("current_slice_snssai"))
            if snssai is None:
                continue
            if snssai_key(snssai) not in allowed_keys:
                hard_conflicts.append(
                    f"QoS-selected slice {(flow.get('allocation') or {}).get('current_slice_snssai')} is not covered by mobility allowedSnssais"
                )

    total_ul = 0.0
    total_dl = 0.0
    for flow in qos_plan.get("target_app", {}).get("flows", []):
        if not isinstance(flow, dict):
            continue
        allocation = flow.get("allocation") if isinstance(flow.get("allocation"), dict) else {}
        total_ul += float(allocation.get("allocated_bandwidth_ul") or 0.0)
        total_dl += float(allocation.get("allocated_bandwidth_dl") or 0.0)
    try:
        ue_ambr_ul = float(str(mobility_draft.policy.ueAmbr.uplink).split()[0]) if mobility_draft.policy.ueAmbr else 0.0
        ue_ambr_dl = float(str(mobility_draft.policy.ueAmbr.downlink).split()[0]) if mobility_draft.policy.ueAmbr else 0.0
    except Exception:
        ue_ambr_ul = 0.0
        ue_ambr_dl = 0.0
    if "ambr_consistency" in problem_config.active_constraints:
        if ue_ambr_ul and ue_ambr_ul < total_ul:
            hard_conflicts.append("UE AMBR uplink is lower than optimizer-assigned uplink bandwidth")
        if ue_ambr_dl and ue_ambr_dl < total_dl:
            hard_conflicts.append("UE AMBR downlink is lower than optimizer-assigned downlink bandwidth")
    if "service_area_consistency" in problem_config.active_constraints and snapshot.userLoc is None and mobility_draft.policy.servAreaRes is not None:
        hard_conflicts.append("service-area restriction requires user location context")
    if not snapshot.presenceAreas and PcfAmPolicyControlRequestTrigger.PRA_CH in (mobility_draft.policy.triggers or []):
        soft_conflicts.append("PRA trigger configured without presence-area context")

    verdicts.append(
        DomainVerdict(
            domain=ControlDomain.MOBILITY,
            status=DomainStatus.REJECTED if hard_conflicts else DomainStatus.APPROVED,
            summary="Cross-domain consistency check between QoS plan and AM policy draft",
            hard_conflicts=hard_conflicts,
            soft_conflicts=soft_conflicts,
            metrics={
                "target_snssai_count": len(target_snssais),
                "allowed_snssai_count": len(allowed_snssais),
                "total_qos_ul": total_ul,
                "total_qos_dl": total_dl,
            },
        )
    )
    return verdicts


__all__ = [
    "build_mobility_draft",
    "build_mobility_snapshot",
    "build_slice_snssai",
    "run_cross_domain_checks",
]
