from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from domain.policy_plan import AssuranceVerdict
from agents.tools.network_graph import NetworkGraph


class AssuranceEvaluator:
    def __init__(self, *, load_snapshot_by_id: Callable[[str], Optional[Dict[str, Any]]]) -> None:
        self.load_snapshot_by_id = load_snapshot_by_id

    def evaluate(self, *, policy: Dict[str, Any], snapshot_id: str, k: float = 0.2) -> AssuranceVerdict:
        policy_id = str(policy.get("policy_id") or "").strip()
        target_type = str(policy.get("target_type") or "").strip()
        flow_id = str(policy.get("flow_id") or "").strip() or None
        supi = str(policy.get("supi") or "").strip()

        if target_type != "flow":
            return AssuranceVerdict(
                policy_id=policy_id,
                flow_id=flow_id,
                status="skipped",
                reason="SLA assurance skipped for non-flow scoped policy.",
                metrics={},
            )

        if not snapshot_id:
            raise RuntimeError(f"policy {policy_id} missing snapshot_id for assurance evaluation")

        snapshot = self.load_snapshot_by_id(snapshot_id)
        if not isinstance(snapshot, dict):
            raise RuntimeError(f"policy {policy_id} snapshot {snapshot_id} not found for assurance evaluation")

        flow = None
        if snapshot.get("nodes") and snapshot.get("edges"):
            graph = NetworkGraph.from_payload(snapshot)
            flow = graph.get_flow_record(supi, flow_id or "")
        else:
            app_data = snapshot.get("apps", [])
            for app in app_data:
                if not isinstance(app, dict):
                    continue
                app_supi = str(app.get("supi") or "").strip()
                for item in app.get("flows", []):
                    if not isinstance(item, dict):
                        continue
                    flow_supi = app_supi or str(item.get("supi") or "").strip()
                    if flow_supi == supi and str(item.get("flow_id") or "").strip() == (flow_id or ""):
                        flow = item
                        break
                if flow is not None:
                    break

        if flow is not None:
            lat_req = float(flow.get("lat") or 0.0)
            jitter_req = float(flow.get("jitter_req") or 0.0)
            gbr_ul = float(flow.get("gbr_ul") or 0.0)
            gbr_dl = float(flow.get("gbr_dl") or 0.0)
            sim_latency = float(flow.get("sim_latency") or 0.0)
            sim_jitter = float(flow.get("sim_jitter") or 0.0)
            sim_throughput_ul = float(flow.get("sim_throughput_ul") or 0.0)
            sim_throughput_dl = float(flow.get("sim_throughput_dl") or 0.0)

            k_lat = (sim_latency - lat_req) / (lat_req if lat_req > 0 else 1.0)
            k_jitter = (sim_jitter - jitter_req) / (jitter_req if jitter_req > 0 else 1.0)
            k_ul = (gbr_ul - sim_throughput_ul) / (gbr_ul if gbr_ul > 0 else 1.0)
            k_dl = (gbr_dl - sim_throughput_dl) / (gbr_dl if gbr_dl > 0 else 1.0)

            metrics = {
                "snapshot_id": str(snapshot.get("snapshot_id") or snapshot_id),
                "latency": sim_latency,
                "latency_requirement": lat_req,
                "jitter": sim_jitter,
                "jitter_requirement": jitter_req,
                "throughput_ul": sim_throughput_ul,
                "throughput_ul_requirement": gbr_ul,
                "throughput_dl": sim_throughput_dl,
                "throughput_dl_requirement": gbr_dl,
            }

            if all(k_i < 0 for k_i in (k_lat, k_jitter, k_ul, k_dl)) or all(k_i <= k for k_i in (k_lat, k_jitter, k_ul, k_dl)):
                return AssuranceVerdict(
                    policy_id=policy_id,
                    flow_id=flow_id,
                    status="satisfied",
                    reason="Observed metrics satisfy flow SLA in the bound snapshot.",
                    metrics=metrics,
                )
            return AssuranceVerdict(
                policy_id=policy_id,
                flow_id=flow_id,
                status="violated",
                reason=f"Flow {flow_id} violates SLA in snapshot {snapshot_id}.",
                metrics=metrics,
            )

        raise RuntimeError(f"policy {policy_id} flow {flow_id or '<unknown>'} not found in snapshot {snapshot_id}")
