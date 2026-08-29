from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from database.models import (
    UeAmPolicyAssociationRecord,
    UeContextRecord,
    UeMobilityEventRecord,
    UeServingNfBindingRecord,
)
from shared.logging import setup_logger

from .flow_catalog import _enrich_pcc_rules_with_flow_catalog
from .session_store import _build_graph_catalogs_for_supi, _get_latest_graph_app_data, session_scope

logger = setup_logger(__name__)

def upsert_ue_context(
    supi: str,
    sm_policy_data: Optional[Dict[str, Any]] = None,
    pcc_rules: Optional[Dict[str, Any]] = None,
    qos_decs: Optional[Dict[str, Any]] = None,
    sess_rules: Optional[Dict[str, Any]] = None,
    traff_cont_decs: Optional[Dict[str, Any]] = None,
    chg_decs: Optional[Dict[str, Any]] = None,
    ursp_rules: Optional[Dict[str, Any]] = None,
    app_catalog: Optional[List[Dict[str, Any]]] = None,
    flow_catalog: Optional[List[Dict[str, Any]]] = None,
    access_mobility_context: Optional[Dict[str, Any]] = None,
    am_policy_context: Optional[Dict[str, Any]] = None,
    serving_nf_context: Optional[Dict[str, Any]] = None,
    mobility_summary: Optional[Dict[str, Any]] = None,
    *,
    session: Session | None = None,
) -> bool:
    """插入或更新UE上下文（关键策略字段版）。"""
    if not supi:
        logger.warning("upsert_ue_context skipped: supi is empty")
        return False

    try:
        if session is None:
            with session_scope() as managed_session:
                return upsert_ue_context(
                    supi=supi, sm_policy_data=sm_policy_data, pcc_rules=pcc_rules, qos_decs=qos_decs,
                    sess_rules=sess_rules, traff_cont_decs=traff_cont_decs, chg_decs=chg_decs,
                    ursp_rules=ursp_rules, app_catalog=app_catalog, flow_catalog=flow_catalog,
                    access_mobility_context=access_mobility_context, am_policy_context=am_policy_context,
                    serving_nf_context=serving_nf_context, mobility_summary=mobility_summary,
                    session=managed_session,
                )
        row = session.query(UeContextRecord).filter(UeContextRecord.supi == supi).first()
        if row is None:
            row = UeContextRecord(
                supi=supi, sm_policy_data=sm_policy_data, pcc_rules=pcc_rules, qos_decs=qos_decs,
                sess_rules=sess_rules, traff_cont_decs=traff_cont_decs, chg_decs=chg_decs,
                ursp_rules=ursp_rules, app_catalog=app_catalog, flow_catalog=flow_catalog,
                access_mobility_context=access_mobility_context, am_policy_context=am_policy_context,
                serving_nf_context=serving_nf_context, mobility_summary=mobility_summary,
            )
            session.add(row)
        else:
            for field, value in {
                "sm_policy_data": sm_policy_data, "pcc_rules": pcc_rules, "qos_decs": qos_decs,
                "sess_rules": sess_rules, "traff_cont_decs": traff_cont_decs, "chg_decs": chg_decs,
                "ursp_rules": ursp_rules, "app_catalog": app_catalog, "flow_catalog": flow_catalog,
                "access_mobility_context": access_mobility_context, "am_policy_context": am_policy_context,
                "serving_nf_context": serving_nf_context, "mobility_summary": mobility_summary,
            }.items():
                if value is not None:
                    setattr(row, field, value)
        return True
    except Exception as e:
        logger.error(f"Failed to upsert UE context for {supi}: {e}")
        return False


def get_ue_context_by_supi(supi: str, snapshot_id: str = "") -> Optional[Dict[str, Any]]:
    """按SUPI读取UE上下文。"""
    if not supi:
        return None

    try:
        derived_app_catalog, derived_flow_catalog = _build_graph_catalogs_for_supi(supi, snapshot_id=snapshot_id)

        with session_scope() as session:
            row = session.query(UeContextRecord).filter(UeContextRecord.supi == supi).first()
            if not row:
                if not derived_app_catalog and not derived_flow_catalog:
                    return None
                return {
                    "supi": supi,
                    "smPolicyData": None,
                    "pccRules": None,
                    "qosDecs": None,
                    "sessRules": None,
                    "traffContDecs": None,
                    "chgDecs": None,
                    "urspRules": None,
                    "app_catalog": derived_app_catalog,
                    "flow_catalog": derived_flow_catalog,
                    "accessMobilityContext": {},
                    "amPolicyContext": {},
                    "servingNfContext": {},
                    "mobilitySummary": {},
                    "created_at": None,
                    "updated_at": None,
                }
            return {
                "supi": row.supi,
                "smPolicyData": row.sm_policy_data,
                "pccRules": row.pcc_rules,
                "qosDecs": row.qos_decs,
                "sessRules": row.sess_rules,
                "traffContDecs": row.traff_cont_decs,
                "chgDecs": row.chg_decs,
                "urspRules": row.ursp_rules,
                "app_catalog": derived_app_catalog,
                "flow_catalog": derived_flow_catalog,
                "accessMobilityContext": row.access_mobility_context or {},
                "amPolicyContext": row.am_policy_context or {},
                "servingNfContext": row.serving_nf_context or {},
                "mobilitySummary": row.mobility_summary or {},
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
    except Exception as e:
        logger.error(f"Failed to get UE context for {supi}: {e}")
        return None


def list_ue_contexts(limit: int = 100) -> List[Dict[str, Any]]:
    """列出UE上下文（按更新时间倒序）。"""
    try:
        with session_scope() as session:
            rows = (
                session.query(UeContextRecord)
                .order_by(UeContextRecord.updated_at.desc())
                .limit(max(1, int(limit)))
                .all()
            )
            return [
                {
                    "supi": row.supi,
                    "pccRules": row.pcc_rules,
                    "qosDecs": row.qos_decs,
                    "urspRules": row.ursp_rules,
                    "app_catalog": _build_graph_catalogs_for_supi(row.supi)[0],
                    "flow_catalog": _build_graph_catalogs_for_supi(row.supi)[1],
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"Failed to list UE contexts: {e}")
        return []


def get_ue_flow_catalog_by_supi(supi: str, snapshot_id: str = "") -> Dict[str, Any]:
    """Return the app/flow catalog for a UE, using snapshot app data as the source of truth."""
    if not supi:
        return {"supi": supi, "app_catalog": [], "flow_catalog": []}

    ctx = get_ue_context_by_supi(supi, snapshot_id=snapshot_id)
    if not ctx:
        return {"supi": supi, "app_catalog": [], "flow_catalog": []}

    return {
        "supi": str(ctx.get("supi") or supi).strip(),
        "app_catalog": ctx.get("app_catalog") or [],
        "flow_catalog": ctx.get("flow_catalog") or [],
    }


AM_POLICY_TYPE = "PcfAmPolicyControlPolicyAssociation"


def _merge_context_bucket(existing_bucket: Any, policy_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    bucket = deepcopy(existing_bucket) if isinstance(existing_bucket, dict) else {}
    bucket[policy_id] = payload
    return bucket


def merge_applied_access_policies(
    existing_context: Dict[str, Any],
    policies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    sm_policy_data = deepcopy(existing_context.get("smPolicyData")) if isinstance(existing_context.get("smPolicyData"), dict) else {}
    pcc_rules = deepcopy(existing_context.get("pccRules")) if isinstance(existing_context.get("pccRules"), dict) else {}
    qos_decs = deepcopy(existing_context.get("qosDecs")) if isinstance(existing_context.get("qosDecs"), dict) else {}
    sess_rules = deepcopy(existing_context.get("sessRules")) if isinstance(existing_context.get("sessRules"), dict) else {}
    traff_cont_decs = deepcopy(existing_context.get("traffContDecs")) if isinstance(existing_context.get("traffContDecs"), dict) else {}
    chg_decs = deepcopy(existing_context.get("chgDecs")) if isinstance(existing_context.get("chgDecs"), dict) else {}
    ursp_rules = deepcopy(existing_context.get("urspRules")) if isinstance(existing_context.get("urspRules"), dict) else {}

    for policy in policies:
        policy_details = policy.get("policy_details")
        if not isinstance(policy_details, dict):
            continue
        policy_id = str(policy.get("policy_id") or policy_details.get("policy_id") or "").strip()
        if not policy_id:
            continue
        policy_type = str(policy.get("policy_type") or "").strip()
        if policy_type == "SmPolicyDecision":
            sm_policy_data[policy_id] = {
                "policy_id": policy_id,
                "policy_type": policy_type,
                "policy_details": policy_details,
            }
            if isinstance(policy_details.get("pccRules"), dict):
                pcc_rules = _merge_context_bucket(pcc_rules, policy_id, policy_details["pccRules"])
            if isinstance(policy_details.get("qosDecs"), dict):
                qos_decs = _merge_context_bucket(qos_decs, policy_id, policy_details["qosDecs"])
            if isinstance(policy_details.get("sessRules"), dict):
                sess_rules = _merge_context_bucket(sess_rules, policy_id, policy_details["sessRules"])
            if isinstance(policy_details.get("traffContDecs"), dict):
                traff_cont_decs = _merge_context_bucket(traff_cont_decs, policy_id, policy_details["traffContDecs"])
            if isinstance(policy_details.get("chgDecs"), dict):
                chg_decs = _merge_context_bucket(chg_decs, policy_id, policy_details["chgDecs"])
        elif policy_type == "UrspRuleRequest":
            ursp_rules[policy_id] = policy_details

    return {
        "sm_policy_data": sm_policy_data,
        "pcc_rules": pcc_rules,
        "qos_decs": qos_decs,
        "sess_rules": sess_rules,
        "traff_cont_decs": traff_cont_decs,
        "chg_decs": chg_decs,
        "ursp_rules": ursp_rules,
    }


def commit_applied_policy_plan(
    *,
    supi: str,
    policies: List[Dict[str, Any]],
    session_id: str,
    snapshot_id: str,
) -> None:
    existing = get_ue_context_by_supi(supi, snapshot_id=snapshot_id) or {}
    current_ue_context = deepcopy(existing)
    access_policies = [
        policy
        for policy in policies
        if str(policy.get("policy_type") or "").strip() != AM_POLICY_TYPE
    ]
    mobility_policies = [
        policy
        for policy in policies
        if str(policy.get("policy_type") or "").strip() == AM_POLICY_TYPE
    ]

    if access_policies:
        merged = merge_applied_access_policies(existing, access_policies)
        catalog = get_ue_flow_catalog_by_supi(supi, snapshot_id=snapshot_id)
        ok = upsert_ue_context(
            supi=supi,
            sm_policy_data=merged["sm_policy_data"],
            pcc_rules=merged["pcc_rules"],
            qos_decs=merged["qos_decs"],
            sess_rules=merged["sess_rules"],
            traff_cont_decs=merged["traff_cont_decs"],
            chg_decs=merged["chg_decs"],
            ursp_rules=merged["ursp_rules"],
            app_catalog=catalog.get("app_catalog") or existing.get("app_catalog") or [],
            flow_catalog=catalog.get("flow_catalog") or existing.get("flow_catalog") or [],
        )
        if not ok:
            raise RuntimeError(f"failed to persist policies for supi {supi}")
        current_ue_context.update(
            {
                "smPolicyData": merged["sm_policy_data"],
                "pccRules": merged["pcc_rules"],
                "qosDecs": merged["qos_decs"],
                "sessRules": merged["sess_rules"],
                "traffContDecs": merged["traff_cont_decs"],
                "chgDecs": merged["chg_decs"],
                "urspRules": merged["ursp_rules"],
                "app_catalog": catalog.get("app_catalog") or existing.get("app_catalog") or [],
                "flow_catalog": catalog.get("flow_catalog") or existing.get("flow_catalog") or [],
            }
        )

    for policy in mobility_policies:
        policy_details = policy.get("policy_details") or {}
        request = policy_details.get("request")
        if not isinstance(request, dict):
            raise RuntimeError(f"AM policy {policy.get('policy_id')} missing request payload")
        association_id = str(policy.get("policy_id") or "").strip()
        if not association_id:
            raise RuntimeError("AM policy missing top-level policy_id")
        persisted = upsert_am_policy_association(
            supi=supi,
            pol_asso_id=association_id,
            association_request=request,
            association_policy=policy_details,
            status="applied",
            trigger_event="JOINT_CONTROL_REEVALUATION",
            session_id=session_id,
            snapshot_id=snapshot_id,
            round_index=1,
        )
        if not persisted:
            raise RuntimeError(f"failed to persist AM policy association for {association_id}")

        am_policy_context = dict(current_ue_context.get("amPolicyContext") or {})
        associations = dict(am_policy_context.get("associations") or {})
        associations[association_id] = policy_details
        am_policy_context["associations"] = associations
        am_policy_context["allowedSnssais"] = request.get("allowedSnssais") or []
        am_policy_context["targetSnssais"] = request.get("targetSnssais") or []
        am_policy_context["mappingSnssais"] = request.get("mappingSnssais") or []
        am_policy_context["rfsp"] = request.get("rfsp")
        am_policy_context["ueAmbr"] = request.get("ueAmbr") or policy_details.get("ueAmbr")
        am_policy_context["ueSliceMbrs"] = policy_details.get("ueSliceMbrs") or []
        am_policy_context["pras"] = policy_details.get("pras") or {}
        access_mobility_context = current_ue_context.get("accessMobilityContext") or {}
        serving_nf_context = current_ue_context.get("servingNfContext") or {}
        mobility_summary = {
            "currentAssociationId": association_id,
            "currentTriggers": policy_details.get("triggers") or [],
            "currentRfsp": request.get("rfsp"),
            "lastUpdatedReason": "policy_dispatch",
        }
        updated = upsert_ue_context(
            supi=supi,
            access_mobility_context=access_mobility_context,
            am_policy_context=am_policy_context,
            serving_nf_context=serving_nf_context,
            mobility_summary=mobility_summary,
        )
        if not updated:
            raise RuntimeError(f"failed to update UE mobility context for {association_id}")
        current_ue_context["amPolicyContext"] = am_policy_context
        current_ue_context["mobilitySummary"] = mobility_summary
        current_ue_context["accessMobilityContext"] = access_mobility_context
        current_ue_context["servingNfContext"] = serving_nf_context

        if serving_nf_context.get("amf_id") or serving_nf_context.get("amf_uri"):
            upsert_serving_nf_binding(
                supi=supi,
                nf_type="AMF",
                nf_instance_id=str(serving_nf_context.get("amf_id") or ""),
                nf_uri=str(serving_nf_context.get("amf_uri") or ""),
                binding_info=serving_nf_context,
            )
        recorded = record_mobility_event(
            supi=supi,
            session_id=session_id,
            snapshot_id=snapshot_id,
            event_type="JOINT_CONTROL_REEVALUATION",
            event_summary="AM policy association updated from policy dispatch execution",
            event_payload={"association_id": association_id, "request": request, "policy": policy_details},
        )
        if not recorded:
            raise RuntimeError(f"failed to record mobility event for {association_id}")


def upsert_am_policy_association(
    *,
    supi: str,
    pol_asso_id: str,
    association_request: Dict[str, Any],
    association_policy: Dict[str, Any],
    status: str,
    trigger_event: str,
    session_id: str = "",
    snapshot_id: str = "",
    round_index: int = 0,
    session: Session | None = None,
) -> bool:
    if not str(supi or "").strip() or not str(pol_asso_id or "").strip():
        raise ValueError("supi and pol_asso_id are required")

    if session is None:
        with session_scope() as managed_session:
            return upsert_am_policy_association(
                supi=supi, pol_asso_id=pol_asso_id, association_request=association_request,
                association_policy=association_policy, status=status, trigger_event=trigger_event,
                session_id=session_id, snapshot_id=snapshot_id, round_index=round_index, session=managed_session,
            )
    session.flush()
    row = (
            session.query(UeAmPolicyAssociationRecord)
            .filter(
                UeAmPolicyAssociationRecord.supi == str(supi).strip(),
                UeAmPolicyAssociationRecord.pol_asso_id == str(pol_asso_id).strip(),
            )
            .first()
    )
    if row is None:
        row = UeAmPolicyAssociationRecord(
                supi=str(supi).strip(),
                pol_asso_id=str(pol_asso_id).strip(),
                session_id=str(session_id or "").strip() or None,
                snapshot_id=str(snapshot_id or "").strip() or None,
                round_index=int(round_index or 0),
                association_request=association_request,
                association_policy=association_policy,
                status=str(status or "draft").strip(),
                trigger_event=str(trigger_event or "").strip() or None,
        )
        session.add(row)
    else:
        row.session_id = str(session_id or "").strip() or None
        row.snapshot_id = str(snapshot_id or "").strip() or None
        row.round_index = int(round_index or 0)
        row.association_request = association_request
        row.association_policy = association_policy
        row.status = str(status or "draft").strip()
        row.trigger_event = str(trigger_event or "").strip() or None
    return True


def list_am_policy_associations_by_supi(supi: str) -> List[Dict[str, Any]]:
    if not str(supi or "").strip():
        return []
    with session_scope() as session:
        rows = (
            session.query(UeAmPolicyAssociationRecord)
            .filter(UeAmPolicyAssociationRecord.supi == str(supi).strip())
            .order_by(UeAmPolicyAssociationRecord.updated_at.desc())
            .all()
        )
        return [
            {
                "supi": row.supi,
                "polAssoId": row.pol_asso_id,
                "session_id": row.session_id,
                "snapshot_id": row.snapshot_id,
                "round_index": row.round_index,
                "request": row.association_request,
                "policy": row.association_policy,
                "status": row.status,
                "trigger_event": row.trigger_event,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in rows
        ]


def record_mobility_event(
    *,
    supi: str,
    event_type: str,
    event_payload: Dict[str, Any],
    event_summary: str = "",
    session_id: str = "",
    snapshot_id: str = "",
    session: Session | None = None,
) -> bool:
    if not str(supi or "").strip() or not str(event_type or "").strip():
        raise ValueError("supi and event_type are required")

    if session is None:
        with session_scope() as managed_session:
            return record_mobility_event(
                supi=supi, event_type=event_type, event_payload=event_payload, event_summary=event_summary,
                session_id=session_id, snapshot_id=snapshot_id, session=managed_session,
            )
    row = UeMobilityEventRecord(
            supi=str(supi).strip(),
            session_id=str(session_id or "").strip() or None,
            snapshot_id=str(snapshot_id or "").strip() or None,
            event_type=str(event_type).strip(),
            event_summary=str(event_summary or "").strip() or None,
            event_payload=event_payload,
    )
    session.add(row)
    return True


def upsert_serving_nf_binding(
    *,
    supi: str,
    nf_type: str,
    nf_instance_id: str = "",
    nf_uri: str = "",
    binding_info: Optional[Dict[str, Any]] = None,
    status: str = "active",
    session: Session | None = None,
) -> bool:
    if not str(supi or "").strip() or not str(nf_type or "").strip():
        raise ValueError("supi and nf_type are required")

    if session is None:
        with session_scope() as managed_session:
            return upsert_serving_nf_binding(
                supi=supi, nf_type=nf_type, nf_instance_id=nf_instance_id, nf_uri=nf_uri,
                binding_info=binding_info, status=status, session=managed_session,
            )
    session.flush()
    row = (
            session.query(UeServingNfBindingRecord)
            .filter(
                UeServingNfBindingRecord.supi == str(supi).strip(),
                UeServingNfBindingRecord.nf_type == str(nf_type).strip(),
            )
            .first()
    )
    if row is None:
        row = UeServingNfBindingRecord(
                supi=str(supi).strip(),
                nf_type=str(nf_type).strip(),
                nf_instance_id=str(nf_instance_id or "").strip() or None,
                nf_uri=str(nf_uri or "").strip() or None,
                binding_info=binding_info or {},
                status=str(status or "active").strip(),
        )
        session.add(row)
    else:
        row.nf_instance_id = str(nf_instance_id or "").strip() or None
        row.nf_uri = str(nf_uri or "").strip() or None
        row.binding_info = binding_info or {}
        row.status = str(status or "active").strip()
    return True



def sync_latest_snapshot_flow_catalog_to_ue_context(snapshot_id: str = "") -> Dict[str, int]:
    """
    Rebuild per-UE app/flow catalogs from the latest graph snapshot and refresh PCC flowInfos
    using five_tuple-derived flowDescription strings.
    """
    app_data = _get_latest_graph_app_data(snapshot_id=snapshot_id)
    if not app_data:
        return {"ues": 0, "flows": 0}

    supis = sorted(
        {
            str(app.get("supi") or "").strip()
            for app in app_data
            if isinstance(app, dict) and str(app.get("supi") or "").strip()
        }
    )

    synced_ues = 0
    synced_flows = 0
    for supi in supis:
        app_catalog, flow_catalog = _build_graph_catalogs_for_supi(supi, snapshot_id=snapshot_id)
        if not app_catalog and not flow_catalog:
            continue

        existing = get_ue_context_by_supi(supi, snapshot_id=snapshot_id) or {}
        enriched_pcc_rules = _enrich_pcc_rules_with_flow_catalog(existing.get("pccRules"), flow_catalog)
        ok = upsert_ue_context(
            supi=supi,
            pcc_rules=enriched_pcc_rules,
            app_catalog=app_catalog,
            flow_catalog=flow_catalog,
        )
        if ok:
            synced_ues += 1
            synced_flows += len(flow_catalog)

    return {"ues": synced_ues, "flows": synced_flows}

