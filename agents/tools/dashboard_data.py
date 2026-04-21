from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import desc

from database.models import (
    AgentTask,
    GraphMetric,
    GraphNode,
    NetworkGraphSnapshot,
    SessionContext,
    UeContextRecord,
)
from agents.tools.db_tool import session_scope


def _isoformat(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC).isoformat()
        return value.isoformat()
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize_flow_status(flow_props: Dict[str, Any]) -> str:
    telemetry = flow_props.get("telemetry") if isinstance(flow_props.get("telemetry"), dict) else {}
    sla = flow_props.get("sla") if isinstance(flow_props.get("sla"), dict) else {}
    latency = _safe_float(telemetry.get("latency") or sla.get("latency"))
    jitter = _safe_float(telemetry.get("jitter") or sla.get("jitter"))
    loss = _safe_float(telemetry.get("loss_rate") or sla.get("loss_rate"))

    if latency is not None and latency <= 20 and (loss is None or loss <= 0.002):
        return "healthy"
    if latency is not None and latency <= 60 and (jitter is None or jitter <= 20):
        return "warning"
    return "critical"


def _derive_topology(payload: Dict[str, Any]) -> Dict[str, Any]:
    nodes = payload.get("nodes", []) or []
    edges = payload.get("edges", []) or []
    metrics = payload.get("metrics", []) or []

    node_by_key = {str(node.get("node_key")): node for node in nodes}
    metrics_by_owner: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        owner_key = str(metric.get("owner_key") or "").strip()
        if owner_key:
            metrics_by_owner[owner_key].append(metric)

    app_to_ue: Dict[str, str] = {}
    flow_to_app: Dict[str, str] = {}
    flow_to_slice: Dict[str, str] = {}
    slice_to_hosts: Dict[str, List[str]] = defaultdict(list)

    for edge in edges:
        edge_type = str(edge.get("edge_type") or "")
        source_key = str(edge.get("source_key") or "")
        target_key = str(edge.get("target_key") or "")
        if edge_type == "owns":
            app_to_ue[target_key] = source_key
        elif edge_type == "contains_flow":
            flow_to_app[target_key] = source_key
        elif edge_type == "served_by_slice":
            flow_to_slice[source_key] = target_key
        elif edge_type == "hosted_on":
            slice_to_hosts[source_key].append(target_key)

    topology_nodes: List[Dict[str, Any]] = []
    for node in nodes:
        node_type = str(node.get("node_type") or "")
        if node_type not in {"ue", "ran_node", "core_node"}:
            continue
        node_key = str(node.get("node_key") or "")
        topology_nodes.append(
            {
                "node_key": node_key,
                "node_type": node_type,
                "label": node.get("label") or node_key,
                "properties": node.get("properties") or {},
                "metrics": metrics_by_owner.get(node_key, []),
            }
        )

    topology_edges: List[Dict[str, Any]] = []
    flow_statuses: List[Dict[str, Any]] = []
    flow_counter = 0
    for node in nodes:
        if str(node.get("node_type") or "") != "flow":
            continue

        flow_key = str(node.get("node_key") or "")
        flow_props = dict(node.get("properties") or {})
        app_key = flow_to_app.get(flow_key, "")
        ue_key = app_to_ue.get(app_key, "")
        slice_key = flow_to_slice.get(flow_key, "")
        hosts = slice_to_hosts.get(slice_key, [])
        ran_hosts = [host for host in hosts if str(node_by_key.get(host, {}).get("node_type") or "") == "ran_node"]
        core_hosts = [host for host in hosts if str(node_by_key.get(host, {}).get("node_type") or "") == "core_node"]
        primary_ran = ran_hosts[0] if ran_hosts else ""
        primary_core = core_hosts[0] if core_hosts else ""
        flow_counter += 1

        if ue_key and primary_ran:
            topology_edges.append(
                {
                    "edge_key": f"{flow_key}:ue-ran",
                    "source_key": ue_key,
                    "target_key": primary_ran,
                    "flow_id": flow_props.get("id"),
                    "service_type": (flow_props.get("service") or {}).get("service_type"),
                    "status": _summarize_flow_status(flow_props),
                }
            )
        if primary_ran and primary_core:
            topology_edges.append(
                {
                    "edge_key": f"{flow_key}:ran-core",
                    "source_key": primary_ran,
                    "target_key": primary_core,
                    "flow_id": flow_props.get("id"),
                    "service_type": (flow_props.get("service") or {}).get("service_type"),
                    "status": _summarize_flow_status(flow_props),
                }
            )

        service = flow_props.get("service") if isinstance(flow_props.get("service"), dict) else {}
        sla = flow_props.get("sla") if isinstance(flow_props.get("sla"), dict) else {}
        allocation = flow_props.get("allocation") if isinstance(flow_props.get("allocation"), dict) else {}
        telemetry = flow_props.get("telemetry") if isinstance(flow_props.get("telemetry"), dict) else {}
        flow_statuses.append(
            {
                "flow_key": flow_key,
                "flow_id": flow_props.get("id"),
                "name": flow_props.get("name"),
                "supi": flow_props.get("supi"),
                "app_id": flow_props.get("app_id"),
                "app_name": flow_props.get("app_name"),
                "service_type": service.get("service_type"),
                "slice": allocation.get("current_slice_snssai"),
                "bw_ul": sla.get("bandwidth_ul"),
                "bw_dl": sla.get("bandwidth_dl"),
                "latency": telemetry.get("latency") or sla.get("latency"),
                "jitter": telemetry.get("jitter") or sla.get("jitter"),
                "throughput_ul": telemetry.get("throughput_ul"),
                "throughput_dl": telemetry.get("throughput_dl"),
                "status": _summarize_flow_status(flow_props),
                "route": {
                    "ue_key": ue_key,
                    "ran_key": primary_ran,
                    "core_key": primary_core,
                },
            }
        )

    return {
        "nodes": topology_nodes,
        "edges": topology_edges,
        "flows": flow_statuses,
        "summary": {
            "ue_count": sum(1 for node in topology_nodes if node["node_type"] == "ue"),
            "an_count": sum(1 for node in topology_nodes if node["node_type"] == "ran_node"),
            "cn_count": sum(1 for node in topology_nodes if node["node_type"] == "core_node"),
            "flow_count": flow_counter,
        },
    }


def _build_graph_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    topology = _derive_topology(payload)
    return {
        "snapshot_id": payload.get("snapshot_id"),
        "trigger_event": payload.get("trigger_event"),
        "updated_at": datetime.now(UTC).isoformat(),
        "raw_graph": payload,
        "topology": topology,
    }


def list_agent_statuses() -> List[Dict[str, Any]]:
    with session_scope() as session:
        task_rows = [
            {
                "task_id": row.task_id,
                "target_agent": row.target_agent,
                "status": row.status,
                "attempts": row.attempts,
                "updated_at": row.updated_at,
                "created_at": row.created_at,
                "last_error": row.last_error,
            }
            for row in session.query(AgentTask).order_by(desc(AgentTask.updated_at), desc(AgentTask.created_at)).all()
        ]

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in task_rows:
        agent_name = str(row["target_agent"] or "").strip()
        if not agent_name:
            continue
        bucket = grouped.setdefault(
            agent_name,
            {
                "agent_name": agent_name,
                "queued": 0,
                "running": 0,
                "succeeded": 0,
                "failed": 0,
                "dead_letter": 0,
                "attempts": 0,
                "last_task_at": None,
                "last_task_id": None,
                "last_error": None,
            },
        )
        status = str(row["status"] or "unknown")
        if status in bucket:
            bucket[status] += 1
        bucket["attempts"] += int(row["attempts"] or 0)
        task_time = row["updated_at"] or row["created_at"]
        if bucket["last_task_at"] is None and task_time is not None:
            bucket["last_task_at"] = _isoformat(task_time)
            bucket["last_task_id"] = row["task_id"]
            bucket["last_error"] = row["last_error"]

    statuses: List[Dict[str, Any]] = []
    for agent in grouped.values():
        if agent["dead_letter"] > 0 or agent["failed"] > 0:
            health = "critical"
        elif agent["running"] > 0:
            health = "active"
        elif agent["queued"] > 0:
            health = "waiting"
        else:
            health = "idle"
        agent["health"] = health
        statuses.append(agent)

    return sorted(statuses, key=lambda item: (item["health"] != "active", item["agent_name"]))


def list_queue_items(limit: int = 100) -> List[Dict[str, Any]]:
    with session_scope() as session:
        rows = [
            {
                "task_id": row.task_id,
                "artifact_id": row.artifact_id,
                "artifact_type": row.artifact_type,
                "source_agent": row.source_agent,
                "target_agent": row.target_agent,
                "session_id": row.session_id,
                "snapshot_id": row.snapshot_id,
                "status": row.status,
                "attempts": row.attempts,
                "max_attempts": row.max_attempts,
                "lease_owner": row.lease_owner,
                "lease_expires_at": row.lease_expires_at,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
                "last_error": row.last_error,
            }
            for row in (
                session.query(AgentTask)
                .order_by(desc(AgentTask.updated_at), desc(AgentTask.created_at))
                .limit(max(1, int(limit)))
                .all()
            )
        ]

    return [
        {
            "task_id": row["task_id"],
            "artifact_id": row["artifact_id"],
            "artifact_type": row["artifact_type"],
            "source_agent": row["source_agent"],
            "target_agent": row["target_agent"],
            "session_id": row["session_id"],
            "snapshot_id": row["snapshot_id"],
            "status": row["status"],
            "attempts": row["attempts"],
            "max_attempts": row["max_attempts"],
            "lease_owner": row["lease_owner"],
            "lease_expires_at": _isoformat(row["lease_expires_at"]),
            "created_at": _isoformat(row["created_at"]),
            "updated_at": _isoformat(row["updated_at"]),
            "last_error": row["last_error"],
        }
        for row in rows
    ]


def list_use_inputs(limit: int = 20) -> List[Dict[str, Any]]:
    with session_scope() as session:
        rows = [
            {
                "session_id": row.session_id,
                "intent_data": row.intent_data,
                "status": row.status,
                "current_stage": row.current_stage,
                "current_step": row.current_step,
                "current_snapshot_id": row.current_snapshot_id,
                "updated_at": row.updated_at,
                "created_at": row.created_at,
            }
            for row in (
                session.query(SessionContext)
                .order_by(desc(SessionContext.updated_at), desc(SessionContext.created_at))
                .limit(max(1, int(limit)))
                .all()
            )
        ]

    records: List[Dict[str, Any]] = []
    for row in rows:
        intent_data = row["intent_data"] if isinstance(row["intent_data"], dict) else {}
        raw_input = intent_data.get("use_input") or intent_data.get("raw_intent") or ""
        if not str(raw_input).strip():
            continue
        records.append(
            {
                "session_id": row["session_id"],
                "text": str(raw_input),
                "status": row["status"],
                "current_stage": row["current_stage"] or row["current_step"],
                "snapshot_id": row["current_snapshot_id"],
                "updated_at": _isoformat(row["updated_at"]),
            }
        )
    return records


def create_use_input_record(text: str, session_id: str = "") -> Dict[str, Any]:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        raise ValueError("text is required")

    payload = {
        "use_input": normalized_text,
        "source": "dashboard",
        "submitted_at": datetime.now(UTC).isoformat(),
    }

    with session_scope() as session:
        row: Optional[SessionContext] = None
        if session_id:
            row = session.query(SessionContext).filter(SessionContext.session_id == session_id).first()

        if row is None:
            row = SessionContext(
                current_step="intent",
                current_stage="intent",
                intent_data=payload,
                status="active",
            )
            session.add(row)
            session.flush()
        else:
            row.current_step = "intent"
            row.current_stage = "intent"
            row.intent_data = payload
            row.status = "active"

        return {
            "session_id": row.session_id,
            "text": normalized_text,
            "status": row.status,
            "submitted_at": payload["submitted_at"],
        }


def get_latest_network_snapshot() -> Optional[Dict[str, Any]]:
    with session_scope() as session:
        row = session.query(NetworkGraphSnapshot).order_by(desc(NetworkGraphSnapshot.created_at)).first()
        if row is None or not isinstance(row.graph_summary, dict):
            return None
        payload = dict(row.graph_summary)
    return _build_graph_response(payload)


def get_node_details(node_key: str) -> Optional[Dict[str, Any]]:
    normalized_key = str(node_key or "").strip()
    if not normalized_key:
        return None

    with session_scope() as session:
        graph_row = session.query(NetworkGraphSnapshot).order_by(desc(NetworkGraphSnapshot.created_at)).first()
        if graph_row is None:
            return None

        node_row = (
            session.query(GraphNode)
            .filter(GraphNode.snapshot_id == graph_row.snapshot_id, GraphNode.node_key == normalized_key)
            .first()
        )
        if node_row is None:
            return None

        snapshot_id = graph_row.snapshot_id
        node_payload = {
            "node_key": node_row.node_key,
            "node_type": node_row.node_type,
            "label": node_row.label,
            "properties": node_row.properties if isinstance(node_row.properties, dict) else {},
        }
        metric_rows = [
            {
                "metric_name": metric.metric_name,
                "metric_value": metric.metric_value,
                "observed_at": metric.observed_at,
            }
            for metric in (
                session.query(GraphMetric)
                .filter(GraphMetric.snapshot_id == snapshot_id, GraphMetric.owner_key == normalized_key)
                .order_by(GraphMetric.metric_name.asc(), GraphMetric.observed_at.desc())
                .all()
            )
        ]

        ue_context = None
        if node_payload["node_type"] == "ue":
            properties = node_payload["properties"]
            supi = str(properties.get("supi") or "").strip()
            if supi:
                ue_context_row = session.query(UeContextRecord).filter(UeContextRecord.supi == supi).first()
                if ue_context_row is not None:
                    ue_context = {
                        "supi": ue_context_row.supi,
                        "app_catalog": ue_context_row.app_catalog,
                        "flow_catalog": ue_context_row.flow_catalog,
                        "updated_at": _isoformat(ue_context_row.updated_at),
                    }

    return {
        "snapshot_id": snapshot_id,
        "node_key": node_payload["node_key"],
        "node_type": node_payload["node_type"],
        "label": node_payload["label"],
        "properties": node_payload["properties"],
        "metrics": [
            {
                "metric_name": metric["metric_name"],
                "metric_value": metric["metric_value"],
                "observed_at": _isoformat(metric["observed_at"]),
            }
            for metric in metric_rows
        ],
        "ue_context": ue_context,
    }


def get_dashboard_overview() -> Dict[str, Any]:
    agent_statuses = list_agent_statuses()
    queue_items = list_queue_items(limit=200)
    use_inputs = list_use_inputs(limit=10)
    latest_graph = get_latest_network_snapshot()

    queue_counter = Counter(item["status"] for item in queue_items)
    return {
        "agents": {
            "total": len(agent_statuses),
            "active": sum(1 for agent in agent_statuses if agent["health"] == "active"),
            "critical": sum(1 for agent in agent_statuses if agent["health"] == "critical"),
            "waiting": sum(1 for agent in agent_statuses if agent["health"] == "waiting"),
        },
        "queue": {
            "queued": queue_counter.get("queued", 0),
            "running": queue_counter.get("running", 0),
            "succeeded": queue_counter.get("succeeded", 0),
            "failed": queue_counter.get("failed", 0),
            "dead_letter": queue_counter.get("dead_letter", 0),
        },
        "use_input_count": len(use_inputs),
        "network": latest_graph["topology"]["summary"] if latest_graph else None,
    }
