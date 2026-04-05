from typing import List, Tuple, Dict, Any, Optional, Union, TYPE_CHECKING
from dataclasses import asdict
from contextlib import contextmanager

from database.connection import SessionLocal
from database.models import NetworkStatusSnapshot, SessionContext, UeContextRecord
import sys
import os
if TYPE_CHECKING:
    from agents.tools.optimizer.models import App, Slice, Node

try:
    from utils.logger import setup_logger
except ImportError:
    # Fallback if running relative
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils.logger import setup_logger

logger = setup_logger(__name__)


def build_flow_description_from_five_tuple(five_tuple: Any) -> Optional[str]:
    """Encode a flow five-tuple into the flowDescription string supported by PCC flowInfos."""
    if not isinstance(five_tuple, (list, tuple)) or len(five_tuple) != 5:
        return None

    src_ip, dst_ip, src_port, dst_port, protocol = five_tuple
    src_ip = str(src_ip or "").strip()
    dst_ip = str(dst_ip or "").strip()
    protocol = str(protocol or "ip").strip().lower() or "ip"

    if not src_ip or not dst_ip:
        return None

    try:
        src_port = int(src_port)
        dst_port = int(dst_port)
    except (TypeError, ValueError):
        return None

    return f"permit out {protocol} from {src_ip} {src_port} to {dst_ip} {dst_port}"


def build_flow_info_from_five_tuple(five_tuple: Any, *, flow_direction: str = "BIDIRECTIONAL") -> Optional[Dict[str, Any]]:
    """Build a FlowInformation-compatible dict from a five tuple."""
    flow_description = build_flow_description_from_five_tuple(five_tuple)
    if not flow_description:
        return None
    return {
        "flowDescription": flow_description,
        "flowDirection": flow_direction,
    }


def _normalize_catalog_flow(app: Dict[str, Any], flow: Dict[str, Any]) -> Dict[str, Any]:
    current_bw_ul = flow.get("old_allocated_bw_ul")
    current_bw_dl = flow.get("old_allocated_bw_dl")
    if current_bw_ul is None:
        current_bw_ul = flow.get("bw_ul")
    if current_bw_dl is None:
        current_bw_dl = flow.get("bw_dl")

    return {
        "supi": app.get("supi"),
        "app_name": app.get("name"),
        "app_id": app.get("app_id"),
        "flow_name": flow.get("name"),
        "flow_id": flow.get("flow_id"),
        "service_type": flow.get("service_type"),
        "service_type_id": flow.get("service_type_id"),
        "bw_ul": flow.get("bw_ul"),
        "bw_dl": flow.get("bw_dl"),
        "gbr_ul": flow.get("gbr_ul"),
        "gbr_dl": flow.get("gbr_dl"),
        "lat": flow.get("lat"),
        "loss_req": flow.get("loss_req"),
        "jitter_req": flow.get("jitter_req"),
        "priority": flow.get("priority"),
        "current_bw_ul": current_bw_ul,
        "current_bw_dl": current_bw_dl,
        "five_tuple": list(flow.get("five_tuple")) if isinstance(flow.get("five_tuple"), (list, tuple)) else None,
    }


def _build_catalogs_from_app_data(app_data: Any, supi: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    app_catalog: List[Dict[str, Any]] = []
    flow_catalog: List[Dict[str, Any]] = []
    if not isinstance(app_data, list) or not supi:
        return app_catalog, flow_catalog

    target_supi = str(supi).strip()
    for app in app_data:
        if not isinstance(app, dict):
            continue
        app_supi = str(app.get("supi") or "").strip()
        if app_supi != target_supi:
            continue

        app_entry = {
            "supi": target_supi,
            "app_name": app.get("name"),
            "app_id": app.get("app_id"),
            "flow_count": len(app.get("flows") or []),
        }
        app_catalog.append(app_entry)

        flows = app.get("flows") or []
        for flow in flows:
            if not isinstance(flow, dict):
                continue
            flow_catalog.append(_normalize_catalog_flow(app, flow))

    return app_catalog, flow_catalog

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _serialize_scenario_for_db(apps: List["App"], slices: List["Slice"], nodes: List["Node"]) -> Dict[str, Any]:
    """Serialize scenario data for DB storage (slice name removed)."""
    apps_data = [asdict(app) for app in apps]
    nodes_data = [asdict(n) for n in nodes]

    slices_data = []
    for s in slices:
        s_dict = asdict(s)
        if "name" in s_dict:
            del s_dict["name"]
        slices_data.append(s_dict)

    return {
        "apps": apps_data,
        "slices": slices_data,
        "nodes": nodes_data
    }


def _normalize_snapshot_payload(
    app_data: Optional[Any],
    slice_data: Optional[Any],
    node_data: Optional[Any],
) -> Dict[str, Any]:
    """将拆分列还原为统一快照字典格式。"""
    return {
        "apps": app_data if isinstance(app_data, list) else [],
        "slices": slice_data if isinstance(slice_data, list) else [],
        "nodes": node_data if isinstance(node_data, list) else [],
    }


def _snapshot_row_to_payload(row: Any) -> Dict[str, Any]:
    return {
        "snapshot_id": str(row.id),
        "timestamp": row.timestamp.isoformat() if getattr(row, "timestamp", None) else None,
        "trigger_event": getattr(row, "trigger_event", None),
        **_normalize_snapshot_payload(
            app_data=getattr(row, "app_data", None),
            slice_data=getattr(row, "slice_data", None),
            node_data=getattr(row, "node_data", None),
        ),
    }

def get_latest_snapshot_data() -> Optional[Dict[str, Any]]:
    """Read latest snapshot payload from the graph snapshot when available."""
    try:
        from agents.tools.network_graph import get_latest_graph

        graph = get_latest_graph()
        if graph is not None:
            payload = graph.to_compatibility_snapshot()
            return {
                "snapshot_id": payload.get("snapshot_id"),
                "apps": payload["apps"],
                "slices": payload["slices"],
                "nodes": payload["nodes"],
            }
    except Exception as e:
        logger.warning(f"Failed to load latest graph-backed snapshot: {e}")

    try:
        with session_scope() as session:
            latest_snapshot = session.query(NetworkStatusSnapshot).order_by(NetworkStatusSnapshot.timestamp.desc()).first()
            if latest_snapshot:
                logger.debug(f"Loaded scenario snapshot (Timestamp: {latest_snapshot.timestamp}).")
                payload = _snapshot_row_to_payload(latest_snapshot)
                return {
                    "apps": payload["apps"],
                    "slices": payload["slices"],
                    "nodes": payload["nodes"],
                }
    except Exception as e:
        logger.warning(f"Failed to load latest snapshot: {e}")
    return None


def get_latest_snapshot_metadata() -> Optional[Dict[str, Any]]:
    """Return metadata for the latest bound graph snapshot when available."""
    try:
        from agents.tools.network_graph import get_latest_graph_snapshot_metadata

        metadata = get_latest_graph_snapshot_metadata()
        if isinstance(metadata, dict):
            return metadata
    except Exception as e:
        logger.warning(f"Failed to load latest graph snapshot metadata: {e}")

    try:
        with session_scope() as session:
            latest_snapshot = session.query(NetworkStatusSnapshot).order_by(NetworkStatusSnapshot.timestamp.desc()).first()
            if not latest_snapshot:
                return None
            return {
                "snapshot_id": str(latest_snapshot.id),
                "timestamp": latest_snapshot.timestamp.isoformat() if latest_snapshot.timestamp else None,
                "trigger_event": latest_snapshot.trigger_event,
            }
    except Exception as e:
        logger.warning(f"Failed to load latest snapshot metadata: {e}")
        return None


def get_snapshot_data_by_id(snapshot_id: Union[str, int]) -> Optional[Dict[str, Any]]:
    """Read a specific graph snapshot payload by snapshot id."""
    normalized_snapshot_id = str(snapshot_id or "").strip()
    if not normalized_snapshot_id:
        return None

    try:
        from agents.tools.network_graph import get_graph_snapshot_payload, NetworkGraph

        payload = get_graph_snapshot_payload(normalized_snapshot_id)
        if isinstance(payload, dict) and payload:
            graph = NetworkGraph.from_payload(payload)
            return graph.to_compatibility_snapshot()
    except Exception as e:
        logger.warning(f"Failed to load graph snapshot {snapshot_id}: {e}")

    try:
        snapshot_id_int = int(normalized_snapshot_id)
    except (TypeError, ValueError):
        logger.warning(f"Invalid snapshot_id: {snapshot_id}")
        return None

    try:
        with session_scope() as session:
            snapshot = session.query(NetworkStatusSnapshot).filter(NetworkStatusSnapshot.id == snapshot_id_int).first()
            if not snapshot:
                return None
            return _snapshot_row_to_payload(snapshot)
    except Exception as e:
        logger.warning(f"Failed to load snapshot {snapshot_id}: {e}")
        return None


def get_latest_session_context(status: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Read the most recently updated session context row.

    Note:
        The current session_context table stores session state, not a full
        turn-by-turn conversation transcript.
    """
    try:
        with session_scope() as session:
            query = session.query(SessionContext)
            if status:
                query = query.filter(SessionContext.status == str(status).strip())

            latest_session = (
                query.order_by(
                    SessionContext.updated_at.desc(),
                    SessionContext.created_at.desc(),
                )
                .first()
            )
            if not latest_session:
                return None

            logger.info(
                "Loaded latest session context "
                f"(session_id={latest_session.session_id}, updated_at={latest_session.updated_at})."
            )
            return {
                "session_id": latest_session.session_id,
                "current_step": latest_session.current_step,
                "intent_data": latest_session.intent_data,
                "policy_data": latest_session.policy_data,
                "status": latest_session.status,
                "created_at": latest_session.created_at.isoformat() if latest_session.created_at else None,
                "updated_at": latest_session.updated_at.isoformat() if latest_session.updated_at else None,
            }
    except Exception as e:
        logger.warning(f"Failed to load latest session context: {e}")
        return None


def create_session_context(
    *,
    current_step: str = "intent",
    intent_data: Optional[Dict[str, Any]] = None,
    policy_data: Optional[Dict[str, Any]] = None,
    status: str = "active",
) -> Optional[str]:
    """Create a session_context row and return the generated session_id."""
    try:
        with session_scope() as session:
            row = SessionContext(
                current_step=current_step,
                current_stage=current_step,
                intent_data=intent_data,
                policy_data=policy_data,
                status=status,
            )
            session.add(row)
            session.flush()
            session_id = str(row.session_id)
            logger.info(f"Created session context: {session_id}")
            return session_id
    except Exception as e:
        logger.warning(f"Failed to create session context: {e}")
        return None


def update_session_context(
    session_id: str,
    *,
    current_step: Optional[str] = None,
    intent_data: Optional[Dict[str, Any]] = None,
    policy_data: Optional[Dict[str, Any]] = None,
    status: Optional[str] = None,
) -> bool:
    """Update an existing session_context row by session_id."""
    if not session_id:
        logger.warning("update_session_context skipped: session_id is empty")
        return False

    try:
        with session_scope() as session:
            row = session.query(SessionContext).filter(SessionContext.session_id == session_id).first()
            if row is None:
                logger.warning(f"update_session_context skipped: session_id not found: {session_id}")
                return False

            if current_step is not None:
                row.current_step = current_step
                row.current_stage = current_step
            if intent_data is not None:
                row.intent_data = intent_data
            if policy_data is not None:
                row.policy_data = policy_data
            if status is not None:
                row.status = status
        return True
    except Exception as e:
        logger.warning(f"Failed to update session context {session_id}: {e}")
        return False


def update_scenario_in_db(apps: List["App"], slices: List["Slice"], nodes: List["Node"], trigger: str = "Optimization-Result") -> bool:
    """
    Persist scenario state as a new NetworkStatusSnapshot.
    Now we treat network state as time-series snapshots instead of just updating a single config row.
    """
    try:
        from uuid import uuid4
        from agents.tools.network_graph import build_and_persist_graph_from_scenario

        snapshot_id = f"graph-{uuid4()}"
        build_and_persist_graph_from_scenario(
            apps,
            slices,
            nodes,
            snapshot_id=snapshot_id,
            trigger_event=trigger,
        )
        return True
    except Exception as e:
        logger.error(f"Failed to save scenario snapshot: {e}")
        return False

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
        return True
    except Exception as e:
        logger.error(f"Failed to upsert UE context for {supi}: {e}")
        return False


def get_ue_context_by_supi(supi: str) -> Optional[Dict[str, Any]]:
    """按SUPI读取UE上下文。"""
    if not supi:
        return None

    try:
        snapshot = get_latest_snapshot_data() or {}
        snapshot_apps = snapshot.get("apps", []) if isinstance(snapshot, dict) else []
        derived_app_catalog, derived_flow_catalog = _build_catalogs_from_app_data(snapshot_apps, supi)

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
                "app_catalog": row.app_catalog if row.app_catalog is not None else derived_app_catalog,
                "flow_catalog": row.flow_catalog if row.flow_catalog is not None else derived_flow_catalog,
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
                    "app_catalog": row.app_catalog,
                    "flow_catalog": row.flow_catalog,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"Failed to list UE contexts: {e}")
        return []


def get_ue_flow_catalog_by_supi(supi: str) -> Dict[str, Any]:
    """Return the app/flow catalog for a UE, using snapshot app data as the source of truth."""
    if not supi:
        return {"supi": supi, "app_catalog": [], "flow_catalog": []}

    ctx = get_ue_context_by_supi(supi)
    if not ctx:
        return {"supi": supi, "app_catalog": [], "flow_catalog": []}

    return {
        "supi": str(ctx.get("supi") or supi).strip(),
        "app_catalog": ctx.get("app_catalog") or [],
        "flow_catalog": ctx.get("flow_catalog") or [],
    }


def _extract_flow_id_from_pcc_rule_id(rule_id: Any) -> Optional[str]:
    text = str(rule_id or "").strip()
    if not text:
        return None
    if text.startswith("pcc-") and len(text) > 4:
        return text[4:]
    if text.startswith("flow-"):
        return text
    return None


def _enrich_pcc_rules_with_flow_catalog(
    pcc_rules: Optional[Dict[str, Any]],
    flow_catalog: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(pcc_rules, dict) or not pcc_rules:
        return pcc_rules

    flow_map = {
        str(flow.get("flow_id")): flow
        for flow in flow_catalog
        if isinstance(flow, dict) and flow.get("flow_id")
    }

    enriched_top: Dict[str, Any] = {}
    for sm_policy_id, rule_map in pcc_rules.items():
        if not isinstance(rule_map, dict):
            enriched_top[sm_policy_id] = rule_map
            continue

        enriched_rule_map: Dict[str, Any] = {}
        for rule_key, rule_obj in rule_map.items():
            if not isinstance(rule_obj, dict):
                enriched_rule_map[rule_key] = rule_obj
                continue

            enriched_rule = dict(rule_obj)
            flow_id = _extract_flow_id_from_pcc_rule_id(enriched_rule.get("pccRuleId") or rule_key)
            flow_entry = flow_map.get(flow_id) if flow_id else None
            if flow_entry:
                flow_info = build_flow_info_from_five_tuple(flow_entry.get("five_tuple"))
                if flow_info:
                    enriched_rule["flowInfos"] = [flow_info]
            enriched_rule_map[rule_key] = enriched_rule

        enriched_top[sm_policy_id] = enriched_rule_map

    return enriched_top


def sync_latest_snapshot_flow_catalog_to_ue_context() -> Dict[str, int]:
    """
    Rebuild per-UE app/flow catalogs from the latest snapshot and refresh PCC flowInfos
    using five_tuple-derived flowDescription strings.
    """
    snapshot = get_latest_snapshot_data() or {}
    app_data = snapshot.get("apps", []) if isinstance(snapshot, dict) else []
    if not isinstance(app_data, list) or not app_data:
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
        app_catalog, flow_catalog = _build_catalogs_from_app_data(app_data, supi)
        if not app_catalog and not flow_catalog:
            continue

        existing = get_ue_context_by_supi(supi) or {}
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
