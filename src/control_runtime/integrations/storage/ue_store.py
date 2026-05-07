from __future__ import annotations

from typing import Any, Dict, List, Optional

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
) -> bool:
    """插入或更新UE上下文（关键策略字段版）。"""
    if not supi:
        logger.warning("upsert_ue_context skipped: supi is empty")
        return False

    try:
        with session_scope() as session:
            row = session.query(UeContextRecord).filter(UeContextRecord.supi == supi).first()
            if row is None:
                row = UeContextRecord(
                    supi=supi,
                    sm_policy_data=sm_policy_data,
                    pcc_rules=pcc_rules,
                    qos_decs=qos_decs,
                    sess_rules=sess_rules,
                    traff_cont_decs=traff_cont_decs,
                    chg_decs=chg_decs,
                    ursp_rules=ursp_rules,
                    app_catalog=app_catalog,
                    flow_catalog=flow_catalog,
                    access_mobility_context=access_mobility_context,
                    am_policy_context=am_policy_context,
                    serving_nf_context=serving_nf_context,
                    mobility_summary=mobility_summary,
                )
                session.add(row)
            else:
                if sm_policy_data is not None:
                    row.sm_policy_data = sm_policy_data
                if pcc_rules is not None:
                    row.pcc_rules = pcc_rules
                if qos_decs is not None:
                    row.qos_decs = qos_decs
                if sess_rules is not None:
                    row.sess_rules = sess_rules
                if traff_cont_decs is not None:
                    row.traff_cont_decs = traff_cont_decs
                if chg_decs is not None:
                    row.chg_decs = chg_decs
                if ursp_rules is not None:
                    row.ursp_rules = ursp_rules
                if app_catalog is not None:
                    row.app_catalog = app_catalog
                if flow_catalog is not None:
                    row.flow_catalog = flow_catalog
                if access_mobility_context is not None:
                    row.access_mobility_context = access_mobility_context
                if am_policy_context is not None:
                    row.am_policy_context = am_policy_context
                if serving_nf_context is not None:
                    row.serving_nf_context = serving_nf_context
                if mobility_summary is not None:
                    row.mobility_summary = mobility_summary
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
) -> bool:
    if not str(supi or "").strip() or not str(pol_asso_id or "").strip():
        raise ValueError("supi and pol_asso_id are required")

    with session_scope() as session:
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
) -> bool:
    if not str(supi or "").strip() or not str(event_type or "").strip():
        raise ValueError("supi and event_type are required")

    with session_scope() as session:
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
) -> bool:
    if not str(supi or "").strip() or not str(nf_type or "").strip():
        raise ValueError("supi and nf_type are required")

    with session_scope() as session:
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

