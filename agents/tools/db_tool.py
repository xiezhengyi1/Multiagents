from typing import List, Tuple, Dict, Any, Optional, Union, TYPE_CHECKING
from contextlib import contextmanager
from difflib import SequenceMatcher
import json
import re

from database.connection import SessionLocal
from database.models import (
    SessionContext,
    UeAmPolicyAssociationRecord,
    UeContextRecord,
    UeMobilityEventRecord,
    UeServingNfBindingRecord,
)
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
    service = flow.get("service") if isinstance(flow.get("service"), dict) else {}
    sla = flow.get("sla") if isinstance(flow.get("sla"), dict) else {}
    allocation = flow.get("allocation") if isinstance(flow.get("allocation"), dict) else {}
    traffic = flow.get("traffic") if isinstance(flow.get("traffic"), dict) else {}

    return {
        "supi": app.get("supi"),
        "app_name": app.get("name"),
        "app_id": app.get("id"),
        "flow_name": flow.get("name"),
        "flow_id": flow.get("id"),
        "dnn": flow.get("dnn") or service.get("dnn"),
        "service": {
            "service_type": service.get("service_type"),
            "service_type_id": service.get("service_type_id"),
            "dnn": service.get("dnn") or flow.get("dnn"),
        },
        "sla": {
            "bandwidth_ul": sla.get("bandwidth_ul"),
            "bandwidth_dl": sla.get("bandwidth_dl"),
            "guaranteed_bandwidth_ul": sla.get("guaranteed_bandwidth_ul"),
            "guaranteed_bandwidth_dl": sla.get("guaranteed_bandwidth_dl"),
            "latency": sla.get("latency"),
            "jitter": sla.get("jitter"),
            "loss_rate": sla.get("loss_rate"),
            "priority": sla.get("priority"),
        },
        "allocation": {
            "current_slice_snssai": allocation.get("current_slice_snssai"),
            "allocated_bandwidth_ul": allocation.get("allocated_bandwidth_ul"),
            "allocated_bandwidth_dl": allocation.get("allocated_bandwidth_dl"),
        },
        "traffic": {
            "packet_size": traffic.get("packet_size"),
            "arrival_rate": traffic.get("arrival_rate"),
            "five_tuple": list(traffic.get("five_tuple")) if isinstance(traffic.get("five_tuple"), (list, tuple)) else None,
        },
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
            "app_id": app.get("id"),
            "flow_count": len(app.get("flows") or []),
        }
        app_catalog.append(app_entry)

        flows = app.get("flows") or []
        for flow in flows:
            if not isinstance(flow, dict):
                continue
            flow_catalog.append(_normalize_catalog_flow(app, flow))

    return app_catalog, flow_catalog


def _normalize_semantic_lookup_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(value or "").strip().lower()).strip()


def _compact_semantic_lookup_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").strip().lower())


def _token_overlap_score(query: str, target: str) -> float:
    query_tokens = {token for token in _normalize_semantic_lookup_text(query).split() if token}
    target_tokens = {token for token in _normalize_semantic_lookup_text(target).split() if token}
    if not query_tokens or not target_tokens:
        return 0.0
    return len(query_tokens & target_tokens) / len(query_tokens)


def _semantic_name_score(query: Any, target: Any) -> float:
    normalized_query = _normalize_semantic_lookup_text(query)
    normalized_target = _normalize_semantic_lookup_text(target)
    if not normalized_query or not normalized_target:
        return 0.0

    compact_query = _compact_semantic_lookup_text(query)
    compact_target = _compact_semantic_lookup_text(target)
    if compact_query and compact_query == compact_target:
        return 1.0

    scores = [
        SequenceMatcher(None, normalized_query, normalized_target).ratio(),
        _token_overlap_score(normalized_query, normalized_target),
    ]
    if compact_query and compact_target and (compact_query in compact_target or compact_target in compact_query):
        scores.append(0.95)
    return max(scores)


def search_flow_targets_by_semantic(
    *,
    app_name: str = "",
    flow_name: str = "",
    limit: int = 5,
    min_score: float = 0.35,
) -> Dict[str, Any]:
    normalized_app_name = str(app_name or "").strip()
    normalized_flow_name = str(flow_name or "").strip()
    if not normalized_app_name and not normalized_flow_name:
        return {
            "query": {"app_name": normalized_app_name, "flow_name": normalized_flow_name},
            "candidate_count": 0,
            "candidates": [],
        }

    app_data = _get_latest_graph_app_data()
    candidates: List[Dict[str, Any]] = []

    for app in app_data:
        if not isinstance(app, dict):
            continue
        flows = app.get("flows") or []
        app_score = _semantic_name_score(normalized_app_name, app.get("name")) if normalized_app_name else 0.0
        for flow in flows:
            if not isinstance(flow, dict):
                continue
            flow_score = _semantic_name_score(normalized_flow_name, flow.get("name")) if normalized_flow_name else 0.0
            if normalized_app_name and normalized_flow_name:
                combined_component_threshold = max(min_score, 0.5)
                if app_score < combined_component_threshold or flow_score < combined_component_threshold:
                    continue
                overall_score = (app_score * 0.45) + (flow_score * 0.55)
            elif normalized_app_name:
                overall_score = app_score
            else:
                overall_score = flow_score
            if overall_score < min_score:
                continue

            candidate = _normalize_catalog_flow(app, flow)
            candidate.update(
                {
                    "match_score": round(overall_score, 4),
                    "app_name_score": round(app_score, 4),
                    "flow_name_score": round(flow_score, 4),
                }
            )
            candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            float(item.get("match_score") or 0.0),
            float(item.get("flow_name_score") or 0.0),
            float(item.get("app_name_score") or 0.0),
            str(item.get("supi") or ""),
            str(item.get("app_id") or ""),
            str(item.get("flow_id") or ""),
        ),
        reverse=True,
    )

    bounded_limit = max(1, int(limit or 5))
    return {
        "query": {"app_name": normalized_app_name, "flow_name": normalized_flow_name},
        "candidate_count": len(candidates),
        "candidates": candidates[:bounded_limit],
    }


def _flatten_semantic_tokens(value: Any) -> List[str]:
    tokens: List[str] = []
    if isinstance(value, dict):
        sst = str(value.get("sst") or "").strip()
        sd = str(value.get("sd") or "").strip()
        if sst or sd:
            combined = "-".join(part for part in (sst, sd) if part)
            normalized = _normalize_semantic_lookup_text(combined)
            if normalized:
                tokens.append(normalized)
        for item in value.values():
            tokens.extend(_flatten_semantic_tokens(item))
        try:
            serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            serialized = str(value)
        normalized = _normalize_semantic_lookup_text(serialized)
        if normalized:
            tokens.append(normalized)
    elif isinstance(value, list):
        for item in value:
            tokens.extend(_flatten_semantic_tokens(item))
    else:
        normalized = _normalize_semantic_lookup_text(value)
        if normalized:
            tokens.append(normalized)
    return list(dict.fromkeys(token for token in tokens if token))


def _match_semantic_query(query: str, tokens: List[str], *, min_score: float = 0.45) -> float:
    normalized_query = _normalize_semantic_lookup_text(query)
    if not normalized_query or not tokens:
        return 0.0

    compact_query = _compact_semantic_lookup_text(query)
    best_score = 0.0
    for token in tokens:
        if not token:
            continue
        score = _semantic_name_score(normalized_query, token)
        compact_token = _compact_semantic_lookup_text(token)
        if compact_query and compact_token and (compact_query in compact_token or compact_token in compact_query):
            score = max(score, 0.95)
        best_score = max(best_score, score)

    return best_score if best_score >= min_score else 0.0


def search_am_policy_targets_by_context(
    *,
    supi: str = "",
    association_id: str = "",
    allowed_snssai: str = "",
    target_snssai: str = "",
    service_area: str = "",
    rfsp: str = "",
    access_type: str = "",
    limit: int = 5,
) -> Dict[str, Any]:
    normalized_supi = str(supi or "").strip()
    normalized_association_id = str(association_id or "").strip()
    normalized_allowed = str(allowed_snssai or "").strip()
    normalized_target = str(target_snssai or "").strip()
    normalized_service_area = str(service_area or "").strip()
    normalized_rfsp = str(rfsp or "").strip()
    normalized_access_type = str(access_type or "").strip()

    query_payload = {
        "supi": normalized_supi,
        "association_id": normalized_association_id,
        "allowed_snssai": normalized_allowed,
        "target_snssai": normalized_target,
        "service_area": normalized_service_area,
        "rfsp": normalized_rfsp,
        "access_type": normalized_access_type,
    }
    if not any(query_payload.values()):
        return {
            "query": query_payload,
            "candidate_count": 0,
            "candidates": [],
        }

    bounded_limit = max(1, int(limit or 5))
    fetch_limit = max(20, bounded_limit * 10)

    try:
        with session_scope() as session:
            query = session.query(UeContextRecord)
            if normalized_supi:
                query = query.filter(UeContextRecord.supi == normalized_supi)
            rows = query.order_by(UeContextRecord.updated_at.desc()).limit(fetch_limit).all()
    except Exception as exc:
        logger.error(f"Failed to search AM policy targets: {exc}")
        return {
            "query": query_payload,
            "candidate_count": 0,
            "candidates": [],
        }

    candidates: List[Dict[str, Any]] = []
    for row in rows:
        am_policy_context = row.am_policy_context or {}
        access_mobility_context = row.access_mobility_context or {}
        mobility_summary = row.mobility_summary or {}

        association_map = am_policy_context.get("associations") or {}
        association_ids = [str(key or "").strip() for key in association_map.keys() if str(key or "").strip()]
        current_association_id = str(mobility_summary.get("currentAssociationId") or "").strip()
        if current_association_id and current_association_id not in association_ids:
            association_ids.append(current_association_id)

        scores: List[float] = []
        match_reasons: List[str] = []

        if normalized_supi:
            scores.append(1.0)
            match_reasons.append("supi")

        if normalized_association_id:
            score = _match_semantic_query(normalized_association_id, association_ids, min_score=0.6)
            if score <= 0.0:
                continue
            scores.append(score)
            match_reasons.append("association_id")

        if normalized_allowed:
            score = _match_semantic_query(
                normalized_allowed,
                _flatten_semantic_tokens(am_policy_context.get("allowedSnssais") or []),
            )
            if score <= 0.0:
                continue
            scores.append(score)
            match_reasons.append("allowed_snssai")

        if normalized_target:
            score = _match_semantic_query(
                normalized_target,
                _flatten_semantic_tokens(am_policy_context.get("targetSnssais") or []),
            )
            if score <= 0.0:
                continue
            scores.append(score)
            match_reasons.append("target_snssai")

        if normalized_service_area:
            service_area_payload = (
                am_policy_context.get("servAreaRes")
                or am_policy_context.get("wlServAreaRes")
                or mobility_summary.get("currentServAreaRes")
                or {}
            )
            score = _match_semantic_query(
                normalized_service_area,
                _flatten_semantic_tokens(service_area_payload),
            )
            if score <= 0.0:
                continue
            scores.append(score)
            match_reasons.append("service_area")

        if normalized_rfsp:
            candidate_rfsp = am_policy_context.get("rfsp")
            if candidate_rfsp is None:
                candidate_rfsp = mobility_summary.get("currentRfsp")
            score = _match_semantic_query(normalized_rfsp, _flatten_semantic_tokens(candidate_rfsp), min_score=0.9)
            if score <= 0.0:
                continue
            scores.append(score)
            match_reasons.append("rfsp")

        if normalized_access_type:
            score = _match_semantic_query(
                normalized_access_type,
                _flatten_semantic_tokens(access_mobility_context.get("accessType")),
                min_score=0.8,
            )
            if score <= 0.0:
                continue
            scores.append(score)
            match_reasons.append("access_type")

        if not scores:
            continue

        candidates.append(
            {
                "supi": row.supi,
                "association_ids": association_ids,
                "allowed_snssais": am_policy_context.get("allowedSnssais") or [],
                "target_snssais": am_policy_context.get("targetSnssais") or [],
                "mapping_snssais": am_policy_context.get("mappingSnssais") or [],
                "rfsp": am_policy_context.get("rfsp") if am_policy_context.get("rfsp") is not None else mobility_summary.get("currentRfsp"),
                "access_type": access_mobility_context.get("accessType"),
                "service_area_restriction": (
                    am_policy_context.get("servAreaRes")
                    or am_policy_context.get("wlServAreaRes")
                    or mobility_summary.get("currentServAreaRes")
                    or {}
                ),
                "current_association_id": current_association_id or None,
                "match_score": round(sum(scores) / len(scores), 4),
                "match_reasons": match_reasons,
            }
        )

    candidates.sort(
        key=lambda item: (
            float(item.get("match_score") or 0.0),
            str(item.get("supi") or ""),
        ),
        reverse=True,
    )

    return {
        "query": query_payload,
        "candidate_count": len(candidates),
        "candidates": candidates[:bounded_limit],
    }

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


def _payload_has_scenario_entities(payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(bool(payload.get(key)) for key in ("apps", "slices", "nodes"))


def _get_latest_graph_snapshot_payload() -> Optional[Dict[str, Any]]:
    try:
        from agents.tools.network_graph import get_latest_graph

        graph = get_latest_graph()
        if graph is None:
            return None
        payload = graph.to_compatibility_snapshot()
        if not _payload_has_scenario_entities(payload):
            return None
        return {
            "snapshot_id": payload.get("snapshot_id"),
            "apps": list(payload.get("apps") or []),
            "slices": list(payload.get("slices") or []),
            "nodes": list(payload.get("nodes") or []),
            "mobility": [],
            "policy_state": {},
        }
    except Exception as exc:
        logger.warning(f"Failed to load latest graph snapshot: {exc}")
        return None


def _get_latest_graph_app_data() -> List[Dict[str, Any]]:
    snapshot = _get_latest_graph_snapshot_payload() or {}
    apps = snapshot.get("apps") if isinstance(snapshot, dict) else []
    return list(apps or []) if isinstance(apps, list) else []


def _build_graph_catalogs_for_supi(supi: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    normalized_supi = str(supi or "").strip()
    if not normalized_supi:
        return [], []

    try:
        from agents.tools.network_graph import get_latest_graph

        graph = get_latest_graph()
        if graph is None:
            return [], []
        snapshot = graph.to_compatibility_snapshot()
        app_data = snapshot.get("apps") if isinstance(snapshot, dict) else []
        if not isinstance(app_data, list):
            return [], []
        return _build_catalogs_from_app_data(app_data, normalized_supi)
    except Exception as exc:
        logger.warning(f"Failed to load graph flow catalog for {normalized_supi}: {exc}")
        return [], []

def get_latest_snapshot_data() -> Optional[Dict[str, Any]]:
    """Read the latest scenario snapshot from graph storage only."""
    return _get_latest_graph_snapshot_payload()


def get_latest_snapshot_metadata() -> Optional[Dict[str, Any]]:
    """Return metadata for the latest scenario snapshot from graph storage only."""
    try:
        from agents.tools.network_graph import get_latest_graph_snapshot_metadata

        metadata = get_latest_graph_snapshot_metadata()
        if isinstance(metadata, dict):
            return metadata
    except Exception as e:
        logger.warning(f"Failed to load latest graph snapshot metadata: {e}")
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
    current_snapshot_id: Optional[str] = None,
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
            if current_snapshot_id is not None:
                row.current_snapshot_id = str(current_snapshot_id or "").strip() or None
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


def update_scenario_in_db(
    apps: List["App"],
    slices: List["Slice"],
    nodes: List["Node"],
    *,
    mobility_data: Optional[List[Dict[str, Any]]] = None,
    policy_data: Optional[Dict[str, Any]] = None,
    trigger: str = "Optimization-Result",
) -> bool:
    """
    Persist scenario state as a new graph snapshot.
    mobility_data / policy_data are accepted for compatibility but no longer written to status snapshots.
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


def get_ue_context_by_supi(supi: str) -> Optional[Dict[str, Any]]:
    """按SUPI读取UE上下文。"""
    if not supi:
        return None

    try:
        derived_app_catalog, derived_flow_catalog = _build_graph_catalogs_for_supi(supi)

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
                traffic = flow_entry.get("traffic") if isinstance(flow_entry.get("traffic"), dict) else {}
                flow_info = build_flow_info_from_five_tuple(traffic.get("five_tuple"))
                if flow_info:
                    enriched_rule["flowInfos"] = [flow_info]
            enriched_rule_map[rule_key] = enriched_rule

        enriched_top[sm_policy_id] = enriched_rule_map

    return enriched_top


def sync_latest_snapshot_flow_catalog_to_ue_context() -> Dict[str, int]:
    """
    Rebuild per-UE app/flow catalogs from the latest graph snapshot and refresh PCC flowInfos
    using five_tuple-derived flowDescription strings.
    """
    app_data = _get_latest_graph_app_data()
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
        app_catalog, flow_catalog = _build_graph_catalogs_for_supi(supi)
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
