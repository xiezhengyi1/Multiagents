from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from database.connection import SessionLocal
from database.models import SessionContext
from shared.logging import setup_logger

from .flow_catalog import _build_catalogs_from_app_data
from ..scenario.common import get_cached_control_scenario

if TYPE_CHECKING:
    from ..optimizer.models import App, Node, Slice

logger = setup_logger(__name__)

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


def _as_payload_list(items: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return result
    for item in items:
        if isinstance(item, dict):
            result.append(item)
        elif is_dataclass(item):
            result.append(asdict(item))
    return result


def _get_latest_graph_snapshot_payload() -> Optional[Dict[str, Any]]:
    cached = get_cached_control_scenario()
    if cached.get("apps") is not None and cached.get("slices") is not None and cached.get("nodes") is not None:
        return {
            "snapshot_id": str(cached.get("snapshot_id") or "").strip(),
            "apps": _as_payload_list(cached.get("apps")),
            "slices": _as_payload_list(cached.get("slices")),
            "nodes": _as_payload_list(cached.get("nodes")),
            "mobility": list(cached.get("mobility") or []),
            "policy_state": dict(cached.get("policy_state") or {}),
        }

    try:
        from ..scenario.network_graph import get_latest_graph

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


def _get_graph_snapshot_payload_for_runtime(snapshot_id: str = "") -> Optional[Dict[str, Any]]:
    normalized_snapshot_id = str(snapshot_id or "").strip()
    if normalized_snapshot_id:
        snapshot = get_snapshot_data_by_id(normalized_snapshot_id)
        if not _payload_has_scenario_entities(snapshot):
            raise RuntimeError(f"Graph snapshot not found or empty: snapshot_id={normalized_snapshot_id}")
        return snapshot
    return _get_latest_graph_snapshot_payload()


def _get_latest_graph_app_data(snapshot_id: str = "") -> List[Dict[str, Any]]:
    cached = get_cached_control_scenario()
    if not str(snapshot_id or "").strip() and cached.get("apps") is not None:
        return _as_payload_list(cached.get("apps"))
    snapshot = _get_graph_snapshot_payload_for_runtime(snapshot_id) or {}
    apps = snapshot.get("apps") if isinstance(snapshot, dict) else []
    return list(apps or []) if isinstance(apps, list) else []


def _build_graph_catalogs_for_supi(supi: str, snapshot_id: str = "") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    normalized_supi = str(supi or "").strip()
    if not normalized_supi:
        return [], []

    try:
        snapshot = _get_graph_snapshot_payload_for_runtime(snapshot_id)
        if not isinstance(snapshot, dict):
            return [], []
        app_data = snapshot.get("apps") if isinstance(snapshot, dict) else []
        if not isinstance(app_data, list):
            return [], []
        return _build_catalogs_from_app_data(app_data, normalized_supi)
    except Exception as exc:
        logger.warning(
            "Failed to load graph flow catalog for %s (snapshot_id=%s): %s",
            normalized_supi,
            str(snapshot_id or "").strip() or "<latest>",
            exc,
        )
        if str(snapshot_id or "").strip():
            raise
        return [], []

def get_latest_snapshot_data() -> Optional[Dict[str, Any]]:
    """Read the latest scenario snapshot from graph storage only."""
    return _get_latest_graph_snapshot_payload()


def get_latest_snapshot_metadata() -> Optional[Dict[str, Any]]:
    """Return metadata for the latest scenario snapshot from graph storage only."""
    cached = get_cached_control_scenario()
    cached_snapshot_id = str(cached.get("snapshot_id") or "").strip()
    if cached_snapshot_id and cached.get("apps") is not None:
        return {
            "snapshot_id": cached_snapshot_id,
            "timestamp": None,
            "trigger_event": "CACHE",
        }
    try:
        from ..scenario.network_graph import get_latest_graph_snapshot_metadata

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
        from ..scenario.network_graph import NetworkGraph, get_graph_snapshot_payload

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
    Graph writes are owned by the simulator/live graph writer.

    Multiagents runtime may only read existing graph snapshots by snapshot_id.
    """
    logger.error("Graph snapshot writes are disabled in Multiagents; refusing trigger=%s", trigger)
    return False

