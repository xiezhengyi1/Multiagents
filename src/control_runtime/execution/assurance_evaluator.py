from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..domain.policy_plan import AssuranceVerdict
from ..integrations.scenario.network_graph import NetworkGraph


QOS_TOLERANCE_RATIO = 0.10


def _within_upper_bound(observed: float, target: float, *, tolerance_ratio: float) -> bool:
    if target <= 0.0:
        return True
    return observed <= target * (1.0 + max(0.0, tolerance_ratio))


def _within_lower_bound(observed: float, target: float, *, tolerance_ratio: float) -> bool:
    if target <= 0.0:
        return True
    return observed >= target * (1.0 - max(0.0, tolerance_ratio))


def _first_float(*values: Any) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


class AssuranceEvaluator:
    def __init__(self, *, load_snapshot_by_id: Callable[[str], Optional[Dict[str, Any]]]) -> None:
        self.load_snapshot_by_id = load_snapshot_by_id

    def evaluate(self, *, policy: Dict[str, Any], snapshot_id: str, k: float = QOS_TOLERANCE_RATIO) -> AssuranceVerdict:
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
            # Compatibility snapshots store flows under apps[].flows[] with `id` and nested SLA/telemetry payloads.
            app_data = snapshot.get("apps", [])
            for app in app_data:
                if not isinstance(app, dict):
                    continue
                app_supi = str(app.get("supi") or "").strip()
                for item in app.get("flows", []):
                    if not isinstance(item, dict):
                        continue
                    flow_supi = app_supi or str(item.get("supi") or "").strip()
                    item_flow_id = str(item.get("flow_id") or item.get("id") or "").strip()
                    if flow_supi == supi and item_flow_id == (flow_id or ""):
                        flow = item
                        break
                if flow is not None:
                    break

        if flow is not None:
            sla = flow.get("sla") if isinstance(flow.get("sla"), dict) else {}
            telemetry = flow.get("telemetry") if isinstance(flow.get("telemetry"), dict) else {}
            lat_req = _first_float(flow.get("lat"), sla.get("latency"))
            jitter_req = _first_float(flow.get("jitter_req"), sla.get("jitter"))
            gbr_ul = _first_float(flow.get("gbr_ul"), sla.get("guaranteed_bandwidth_ul"))
            gbr_dl = _first_float(flow.get("gbr_dl"), sla.get("guaranteed_bandwidth_dl"))
            sim_latency = _first_float(flow.get("sim_latency"), telemetry.get("latency"))
            sim_jitter = _first_float(flow.get("sim_jitter"), telemetry.get("jitter"))
            sim_throughput_ul = _first_float(flow.get("sim_throughput_ul"), telemetry.get("throughput_ul"))
            sim_throughput_dl = _first_float(flow.get("sim_throughput_dl"), telemetry.get("throughput_dl"))
            tolerance_ratio = max(0.0, float(k))
            latency_ok = _within_upper_bound(sim_latency, lat_req, tolerance_ratio=tolerance_ratio)
            jitter_ok = _within_upper_bound(sim_jitter, jitter_req, tolerance_ratio=tolerance_ratio)
            throughput_ul_ok = _within_lower_bound(sim_throughput_ul, gbr_ul, tolerance_ratio=tolerance_ratio)
            throughput_dl_ok = _within_lower_bound(sim_throughput_dl, gbr_dl, tolerance_ratio=tolerance_ratio)

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
                "tolerance_ratio": tolerance_ratio,
                "latency_within_tolerance": latency_ok,
                "jitter_within_tolerance": jitter_ok,
                "throughput_ul_within_tolerance": throughput_ul_ok,
                "throughput_dl_within_tolerance": throughput_dl_ok,
            }

            if latency_ok and jitter_ok and throughput_ul_ok and throughput_dl_ok:
                return AssuranceVerdict(
                    policy_id=policy_id,
                    flow_id=flow_id,
                    status="satisfied",
                    reason="Observed metrics satisfy flow SLA within the configured tolerance.",
                    metrics=metrics,
                )
            return AssuranceVerdict(
                policy_id=policy_id,
                flow_id=flow_id,
                status="violated",
                reason=f"Flow {flow_id} violates SLA beyond the configured tolerance in snapshot {snapshot_id}.",
                metrics=metrics,
            )

        raise RuntimeError(f"policy {policy_id} flow {flow_id or '<unknown>'} not found in snapshot {snapshot_id}")
