from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

from database.models import GraphEdge, GraphMetric, GraphNode, NetworkGraphSnapshot
from shared.logging import setup_logger
from ..storage import session_scope


logger = setup_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _join_key_parts(*parts: Any) -> str:
    return ":".join(str(part).strip() for part in parts if str(part or "").strip())


def _app_identity(properties: Dict[str, Any], fallback: str = "") -> str:
    supi = str(properties.get("supi") or "").strip()
    app_id = str(properties.get("id") or "").strip()
    if supi and app_id:
        return f"{supi}:{app_id}"
    if app_id:
        return app_id
    return fallback


class NetworkGraph:
    NODE_TYPES = {"ue", "app", "flow", "session", "slice", "ran_node", "core_node", "policy_binding"}
    EDGE_TYPES = {
        "owns",
        "contains_flow",
        "served_by_slice",
        "hosted_on",
        "connected_to",
        "governed_by_policy",
        "runs_on_session",
        "uses_slice",
        "uses_session",
    }

    def __init__(self, *, snapshot_id: str = "", trigger_event: str = "") -> None:
        self.snapshot_id = str(snapshot_id or "").strip()
        self.trigger_event = str(trigger_event or "").strip()
        self.graph = nx.MultiDiGraph()

    def add_node(self, node_key: str, node_type: str, *, label: str = "", properties: Optional[Dict[str, Any]] = None) -> None:
        normalized_type = str(node_type or "").strip()
        if normalized_type not in self.NODE_TYPES:
            raise ValueError(f"Unsupported node_type: {node_type}")
        self.graph.add_node(
            str(node_key),
            node_type=normalized_type,
            label=str(label or ""),
            properties=dict(properties or {}),
        )

    def add_edge(
        self,
        source_key: str,
        target_key: str,
        edge_type: str,
        *,
        edge_key: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        normalized_type = str(edge_type or "").strip()
        if normalized_type not in self.EDGE_TYPES:
            raise ValueError(f"Unsupported edge_type: {edge_type}")
        self.graph.add_edge(
            str(source_key),
            str(target_key),
            key=str(edge_key),
            edge_type=normalized_type,
            properties=dict(properties or {}),
        )

    def upsert_metric(
        self,
        owner_type: str,
        owner_key: str,
        metric_name: str,
        metric_value: Any,
        *,
        observed_at: Optional[str] = None,
    ) -> None:
        metric_bucket = self.graph.graph.setdefault("metrics", [])
        metric_bucket.append(
            {
                "owner_type": str(owner_type),
                "owner_key": str(owner_key),
                "metric_name": str(metric_name),
                "metric_value": metric_value,
                "observed_at": str(observed_at or _utcnow_iso()),
            }
        )

    def to_payload(self) -> Dict[str, Any]:
        nodes = []
        for node_key, attrs in self.graph.nodes(data=True):
            nodes.append(
                {
                    "node_key": str(node_key),
                    "node_type": attrs.get("node_type"),
                    "label": attrs.get("label"),
                    "properties": attrs.get("properties") or {},
                }
            )

        edges = []
        for source, target, edge_key, attrs in self.graph.edges(keys=True, data=True):
            edges.append(
                {
                    "edge_key": str(edge_key),
                    "edge_type": attrs.get("edge_type"),
                    "source_key": str(source),
                    "target_key": str(target),
                    "properties": attrs.get("properties") or {},
                }
            )

        return {
            "snapshot_id": self.snapshot_id,
            "trigger_event": self.trigger_event,
            "nodes": nodes,
            "edges": edges,
            "metrics": list(self.graph.graph.get("metrics", [])),
        }

    def to_summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for _node_key, attrs in self.graph.nodes(data=True):
            node_type = str(attrs.get("node_type") or "unknown")
            counts[node_type] = counts.get(node_type, 0) + 1
        return {
            "snapshot_id": self.snapshot_id,
            "trigger_event": self.trigger_event,
            "node_counts": counts,
            "edge_count": self.graph.number_of_edges(),
            "metric_count": len(self.graph.graph.get("metrics", [])),
        }

    def to_compatibility_snapshot(self) -> Dict[str, Any]:
        apps: Dict[str, Dict[str, Any]] = {}
        slices: List[Dict[str, Any]] = []
        nodes: List[Dict[str, Any]] = []

        for node_key, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("node_type")
            properties = dict(attrs.get("properties") or {})
            if node_type == "app":
                app_bucket_key = _app_identity(properties, fallback=str(node_key))
                app_payload = dict(properties)
                app_payload["flows"] = []
                apps[app_bucket_key] = app_payload
            elif node_type == "slice":
                slices.append(properties)
            elif node_type in {"ran_node", "core_node"}:
                nodes.append(properties)

        for source, target, _edge_key, attrs in self.graph.edges(keys=True, data=True):
            edge_type = attrs.get("edge_type")
            if edge_type != "contains_flow":
                continue
            app_props = self.graph.nodes[source].get("properties") if source in self.graph.nodes else None
            flow_props = self.graph.nodes[target].get("properties") if target in self.graph.nodes else None
            if not isinstance(app_props, dict) or not isinstance(flow_props, dict):
                continue
            app_bucket_key = _app_identity(app_props, fallback=str(source))
            apps.setdefault(app_bucket_key, dict(app_props))
            apps[app_bucket_key].setdefault("flows", [])
            apps[app_bucket_key]["flows"].append(dict(flow_props))

        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": _utcnow_iso(),
            "trigger_event": self.trigger_event,
            "apps": list(apps.values()),
            "slices": slices,
            "nodes": nodes,
        }

    def get_flow_record(self, supi: str, flow_id: str) -> Optional[Dict[str, Any]]:
        target_supi = str(supi or "").strip()
        target_flow = str(flow_id or "").strip()
        for _node_key, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") != "flow":
                continue
            properties = attrs.get("properties") or {}
            if str(properties.get("supi") or "").strip() == target_supi and str(properties.get("id") or "").strip() == target_flow:
                return dict(properties)
        return None

    def build_flow_catalog(self, supi: str) -> Dict[str, Any]:
        target_supi = str(supi or "").strip()
        app_catalog: Dict[str, Dict[str, Any]] = {}
        flow_catalog: List[Dict[str, Any]] = []

        for _node_key, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("node_type")
            properties = attrs.get("properties") or {}
            if node_type == "app" and str(properties.get("supi") or "").strip() == target_supi:
                app_id = str(properties.get("id") or "")
                app_bucket_key = _app_identity(properties, fallback=str(_node_key))
                app_catalog[app_bucket_key] = {
                    "supi": target_supi,
                    "app_name": properties.get("name"),
                    "app_id": app_id,
                    "flow_count": 0,
                }
            if node_type == "flow" and str(properties.get("supi") or "").strip() == target_supi:
                flow_catalog.append(dict(properties))
                app_id = str(properties.get("app_id") or "")
                app_bucket_key = f"{target_supi}:{app_id}" if target_supi and app_id else str(app_id or _node_key)
                if app_id:
                    app_catalog.setdefault(
                        app_bucket_key,
                        {
                            "supi": target_supi,
                            "app_name": properties.get("app_name"),
                            "app_id": app_id,
                            "flow_count": 0,
                        },
                    )
                    app_catalog[app_bucket_key]["flow_count"] += 1

        return {
            "supi": target_supi,
            "app_catalog": list(app_catalog.values()),
            "flow_catalog": flow_catalog,
        }

    @classmethod
    def from_scenario(
        cls,
        apps: Iterable[Any],
        slices: Iterable[Any],
        nodes: Iterable[Any],
        *,
        snapshot_id: str = "",
        trigger_event: str = "",
    ) -> "NetworkGraph":
        graph = cls(snapshot_id=snapshot_id, trigger_event=trigger_event)

        for app in apps:
            app_payload = asdict(app) if not isinstance(app, dict) else dict(app)
            supi = str(app_payload.get("supi") or "").strip()
            app_id = str(app_payload.get("id") or app_payload.get("name") or "")
            app_key = _join_key_parts("app", supi, app_id)
            if supi:
                ue_key = _join_key_parts("ue", supi)
                graph.add_node(ue_key, "ue", label=supi, properties={"supi": supi})
                graph.add_node(app_key, "app", label=str(app_payload.get("name") or app_id), properties=app_payload)
                graph.add_edge(ue_key, app_key, "owns", edge_key=f"{ue_key}->{app_key}", properties={"supi": supi})
            else:
                graph.add_node(app_key, "app", label=str(app_payload.get("name") or app_id), properties=app_payload)

            for flow in app_payload.get("flows", []) or []:
                flow_payload = dict(flow)
                flow_payload["app_id"] = app_id
                flow_payload["app_name"] = app_payload.get("name")
                flow_payload["supi"] = supi
                flow_id = str(flow_payload.get("id") or "").strip()
                flow_key = _join_key_parts("flow", supi, app_id, flow_id)
                graph.add_node(flow_key, "flow", label=str(flow_payload.get("name") or flow_payload.get("id") or ""), properties=flow_payload)
                graph.add_edge(app_key, flow_key, "contains_flow", edge_key=f"{app_key}->{flow_key}", properties={"app_id": app_id})
                allocation = flow_payload.get("allocation") if isinstance(flow_payload.get("allocation"), dict) else {}
                sla = flow_payload.get("sla") if isinstance(flow_payload.get("sla"), dict) else {}
                telemetry = flow_payload.get("telemetry") if isinstance(flow_payload.get("telemetry"), dict) else {}
                current_slice = str(allocation.get("current_slice_snssai") or "").strip()
                if current_slice:
                    slice_key = _join_key_parts("slice", current_slice)
                    graph.add_edge(flow_key, slice_key, "served_by_slice", edge_key=f"{flow_key}->{slice_key}", properties={"slice": current_slice})
                for metric_name, metric_value in (
                    ("sla.bandwidth_ul", sla.get("bandwidth_ul")),
                    ("sla.bandwidth_dl", sla.get("bandwidth_dl")),
                    ("sla.guaranteed_bandwidth_ul", sla.get("guaranteed_bandwidth_ul")),
                    ("sla.guaranteed_bandwidth_dl", sla.get("guaranteed_bandwidth_dl")),
                    ("sla.latency", sla.get("latency")),
                    ("sla.jitter", sla.get("jitter")),
                    ("sla.loss_rate", sla.get("loss_rate")),
                    ("telemetry.latency", telemetry.get("latency")),
                    ("telemetry.jitter", telemetry.get("jitter")),
                    ("telemetry.throughput_ul", telemetry.get("throughput_ul")),
                    ("telemetry.throughput_dl", telemetry.get("throughput_dl")),
                ):
                    if metric_value is not None:
                        graph.upsert_metric("node", flow_key, metric_name, metric_value)

        for slice_obj in slices:
            slice_payload = asdict(slice_obj) if not isinstance(slice_obj, dict) else dict(slice_obj)
            snssai = str(slice_payload.get("snssai") or f"{int(slice_payload.get('sst', 0)):02X}{slice_payload.get('sd', '000000')}")
            slice_payload["snssai"] = snssai
            slice_key = _join_key_parts("slice", snssai)
            graph.add_node(slice_key, "slice", label=str(slice_payload.get("name") or snssai), properties=slice_payload)
            capacity = slice_payload.get("capacity") if isinstance(slice_payload.get("capacity"), dict) else {}
            load = slice_payload.get("load") if isinstance(slice_payload.get("load"), dict) else {}
            qos = slice_payload.get("qos") if isinstance(slice_payload.get("qos"), dict) else {}
            for metric_name, metric_value in (
                ("capacity.total_bandwidth_ul", capacity.get("total_bandwidth_ul")),
                ("capacity.total_bandwidth_dl", capacity.get("total_bandwidth_dl")),
                ("capacity.guaranteed_bandwidth_ul", capacity.get("guaranteed_bandwidth_ul")),
                ("capacity.guaranteed_bandwidth_dl", capacity.get("guaranteed_bandwidth_dl")),
                ("load.current_bandwidth_ul", load.get("current_bandwidth_ul")),
                ("load.current_bandwidth_dl", load.get("current_bandwidth_dl")),
                ("qos.latency", qos.get("latency")),
                ("qos.processing_delay", qos.get("processing_delay")),
                ("qos.loss_rate", qos.get("loss_rate")),
                ("qos.jitter", qos.get("jitter")),
            ):
                if metric_value is not None:
                    graph.upsert_metric("node", slice_key, metric_name, metric_value)

        for node_obj in nodes:
            node_payload = asdict(node_obj) if not isinstance(node_obj, dict) else dict(node_obj)
            node_name = str(node_payload.get("name") or node_payload.get("id") or "")
            node_type = "ran_node" if str(node_payload.get("node_type") or "").upper() == "AN" else "core_node"
            node_key = _join_key_parts(node_type, node_name)
            graph.add_node(node_key, node_type, label=node_name, properties=node_payload)
            for hosted_slice in node_payload.get("hosted_slice_snssais", []) or []:
                slice_key = _join_key_parts("slice", hosted_slice)
                graph.add_edge(slice_key, node_key, "hosted_on", edge_key=f"{slice_key}->{node_key}", properties={"hosted": True})
            capacity = node_payload.get("capacity") if isinstance(node_payload.get("capacity"), dict) else {}
            for metric_name, metric_value in (
                ("capacity.cpu", capacity.get("cpu")),
                ("capacity.memory", capacity.get("memory")),
                ("capacity.mec", capacity.get("mec")),
                ("capacity.prb", capacity.get("prb")),
            ):
                if metric_value is not None:
                    graph.upsert_metric("node", node_key, metric_name, metric_value)

        return graph

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "NetworkGraph":
        graph = cls(snapshot_id=str(payload.get("snapshot_id") or ""), trigger_event=str(payload.get("trigger_event") or ""))
        for node in payload.get("nodes", []) or []:
            graph.add_node(
                str(node.get("node_key") or ""),
                str(node.get("node_type") or ""),
                label=str(node.get("label") or ""),
                properties=dict(node.get("properties") or {}),
            )
        for edge in payload.get("edges", []) or []:
            graph.add_edge(
                str(edge.get("source_key") or ""),
                str(edge.get("target_key") or ""),
                str(edge.get("edge_type") or ""),
                edge_key=str(edge.get("edge_key") or ""),
                properties=dict(edge.get("properties") or {}),
            )
        for metric in payload.get("metrics", []) or []:
            graph.upsert_metric(
                metric.get("owner_type"),
                metric.get("owner_key"),
                metric.get("metric_name"),
                metric.get("metric_value"),
                observed_at=metric.get("observed_at"),
            )
        return graph


def persist_network_graph(graph: NetworkGraph, *, base_network_snapshot_id: str = "") -> str:
    raise RuntimeError("Graph snapshot writes are disabled in Multiagents; use the simulator/live graph writer.")


def build_and_persist_graph_from_scenario(
    apps: Iterable[Any],
    slices: Iterable[Any],
    nodes: Iterable[Any],
    *,
    snapshot_id: str,
    trigger_event: str,
    base_network_snapshot_id: str = "",
) -> str:
    raise RuntimeError("Graph snapshot creation is disabled in Multiagents; read an existing snapshot_id instead.")


def get_graph_snapshot_payload(snapshot_id: str) -> Optional[Dict[str, Any]]:
    if not snapshot_id:
        return None
    try:
        with session_scope() as session:
            row = session.query(NetworkGraphSnapshot).filter(NetworkGraphSnapshot.snapshot_id == snapshot_id).first()
            if row is None:
                return None
            return dict(row.graph_summary or {})
    except Exception as exc:
        logger.warning("Failed to load graph snapshot %s: %s", snapshot_id, exc)
        return None


def get_latest_graph_snapshot_metadata() -> Optional[Dict[str, Any]]:
    try:
        with session_scope() as session:
            row = session.query(NetworkGraphSnapshot).order_by(NetworkGraphSnapshot.created_at.desc()).first()
            if row is None:
                return None
            return {
                "snapshot_id": row.snapshot_id,
                "timestamp": row.created_at.isoformat() if row.created_at else None,
                "trigger_event": row.trigger_event,
            }
    except Exception as exc:
        logger.warning("Failed to load latest graph snapshot metadata: %s", exc)
        return None


def get_latest_graph() -> Optional[NetworkGraph]:
    meta = get_latest_graph_snapshot_metadata()
    if not meta:
        return None
    payload = get_graph_snapshot_payload(str(meta.get("snapshot_id") or ""))
    if not isinstance(payload, dict):
        return None
    return NetworkGraph.from_payload(payload)

