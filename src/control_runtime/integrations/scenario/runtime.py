from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from shared.logging import setup_logger

from ..storage import get_latest_snapshot_data
from ..optimizer.models import App, Flow, Node, NodeCapacity, Slice, SliceCapacity, SliceLoad, SliceQos
from .common import (
    DEFAULT_SCENARIO_APPS_JSON,
    _build_flow_from_dict,
    _extract_app_supi,
    _normalize_or_generate_id,
    cache_scenario,
    get_cached_control_scenario,
    get_cached_scenario,
    snapshot_uses_new_schema,
)
from .ue_bootstrap import (
    _build_mobility_snapshot_payload,
    _build_policy_state_payload,
    _load_latest_graph_scenario_strict,
    _seed_ue_contexts_from_apps,
    rebuild_ue_related_tables_from_graph_snapshot,
    rebuild_ue_related_tables_from_latest_graph,
    sync_latest_flow_five_tuples_to_ue_context,
)
from .yaml_loader import _create_scenario_from_yaml, _deserialize_scenario

logger = setup_logger(__name__)

def _snapshot_uses_new_schema(snapshot_data: Optional[Dict[str, Any]]) -> bool:
    return snapshot_uses_new_schema(snapshot_data)


def _create_default_scenario() -> Tuple[List[App], List[Slice], List[Node]]:
    """Generate deterministic default scenario:
    - 5 AN
    - 4 CN (UPF)
    - 10 APP
    - each APP has 1~2 flows
    - 5 slices
    """

    rng = random.Random(20260315)

    slices_data = [
        Slice("S1_Gold", sst=2, sd="000001", capacity=SliceCapacity(total_bandwidth_ul=120, total_bandwidth_dl=120), load=SliceLoad(), qos=SliceQos(latency=3, processing_delay=1, loss_rate=0.001, jitter=1.5)),
        Slice("S2_Silver", sst=1, sd="000001", capacity=SliceCapacity(total_bandwidth_ul=220, total_bandwidth_dl=220), load=SliceLoad(), qos=SliceQos(latency=10, processing_delay=2, loss_rate=0.01, jitter=8)),
        Slice("S3_Public", sst=1, sd="000002", capacity=SliceCapacity(total_bandwidth_ul=180, total_bandwidth_dl=180), load=SliceLoad(), qos=SliceQos(latency=40, processing_delay=5, loss_rate=0.03, jitter=25)),
        Slice("S4_Platinum", sst=2, sd="000002", capacity=SliceCapacity(total_bandwidth_ul=100, total_bandwidth_dl=80), load=SliceLoad(), qos=SliceQos(latency=2, processing_delay=0.8, loss_rate=0.0005, jitter=1.0)),
        Slice("S5_Massive", sst=3, sd="000001", capacity=SliceCapacity(total_bandwidth_ul=160, total_bandwidth_dl=160), load=SliceLoad(), qos=SliceQos(latency=100, processing_delay=10, loss_rate=0.05, jitter=60)),
    ]

    slice_by_snssai = {f"{s.sst:02X}{s.sd}": s for s in slices_data}
    all_slice_snssai = [f"{s.sst:02X}{s.sd}" for s in slices_data]
    
    # 关键步骤：保证“节点托管切片并集”覆盖全部切片，且 MEC 在节点间分布
    def _build_hosted_slices(node_idx: int, node_count: int, target_k: int) -> List[str]:
        hosted = {
            all_slice_snssai[node_idx % len(all_slice_snssai)],
            all_slice_snssai[(node_idx + 1) % len(all_slice_snssai)],
        }
        remain = [n for n in all_slice_snssai if n not in hosted]
        extra_k = max(0, min(target_k - len(hosted), len(remain)))
        if extra_k > 0:
            hosted.update(rng.sample(remain, k=extra_k))
        return sorted(hosted)

    nodes_data: List[Node] = []
    for i in range(5):
        hosted = _build_hosted_slices(node_idx=i, node_count=5, target_k=3)
        # uRLLC(sst=2) 在 AN 侧更偏 MEC，下式体现“不同 AN 节点的 MEC 分布差异”
        an_pref_count = sum(1 for n in hosted if slice_by_snssai[n].sst == 2)
        nodes_data.append(
            Node(
                id=i,
                name=f"AN_gNB_{i}",
                node_type="AN",
                capacity=NodeCapacity(
                    cpu=1200 + i * 10,
                    memory=250 + i * 32,
                    mec=1200 + an_pref_count * 45 + i * 8,
                    prb=2500 + i * 50,
                ),
                hosted_slice_snssais=hosted,
            )
        )

    for i in range(4):
        hosted = _build_hosted_slices(node_idx=i, node_count=4, target_k=3)
        # eMBB/mMTC(sst=1/3) 在 CN 侧更偏 MEC，下式体现“不同 CN 节点的 MEC 分布差异”
        cn_pref_count = sum(1 for n in hosted if slice_by_snssai[n].sst in (1, 3))
        nodes_data.append(
            Node(
                id=100 + i,
                name=f"CN_UPF_{i}",
                node_type="CN",
                capacity=NodeCapacity(
                    cpu=4200 + i * 60,
                    memory=2024 + i * 128,
                    mec=1800 + cn_pref_count * 35 + i * 15,
                    prb=0,
                ),
                hosted_slice_snssais=hosted,
            )
        )

    apps_data: List[App] = []
    used_suffixes: set = set()
    
    # 使用预定义的 DEFAULT_SCENARIO_APPS_JSON 替换随机生成
    for app_dict in DEFAULT_SCENARIO_APPS_JSON:
        flows: List[Flow] = []
        for f_dict in app_dict.get("flows", []):
            flows.append(_build_flow_from_dict(f_dict, used_suffixes=used_suffixes))

        apps_data.append(
            App(
                name=app_dict["name"],
                id=_normalize_or_generate_id(
                    app_dict.get("id", app_dict.get("app_id")),
                    "app",
                    used_suffixes,
                ),
                supi=_extract_app_supi(app_dict),
                flows=flows,
            )
        )

    return apps_data, slices_data, nodes_data


def get_initial_scenario() -> Tuple[List[App], List[Slice], List[Node]]:
    """Load the current scenario from an existing graph snapshot."""
    snapshot_data = get_latest_snapshot_data()
    if snapshot_data:
        if _snapshot_uses_new_schema(snapshot_data):
            apps, slices, nodes = _deserialize_scenario(snapshot_data)
            mobility_payload = snapshot_data.get("mobility") if isinstance(snapshot_data, dict) else None
            policy_payload = snapshot_data.get("policy_state") if isinstance(snapshot_data, dict) else None
            seeded = _seed_ue_contexts_from_apps(apps)
            sync_summary = sync_latest_flow_five_tuples_to_ue_context()
            print(f"[UEContext] seeded {seeded} UE records (from snapshot)")
            print(f"[UEContext] synced five-tuples for {sync_summary['ues']} UEs / {sync_summary['flows']} flows (from snapshot)")
            cache_scenario(
                apps,
                slices,
                nodes,
                mobility_payload or _build_mobility_snapshot_payload(apps),
                policy_payload or _build_policy_state_payload(apps),
                snapshot_id=str(snapshot_data.get("snapshot_id") or "").strip(),
            )
            return apps, slices, nodes
        raise RuntimeError("latest graph snapshot uses an unsupported schema; Multiagents will not reset or rewrite graph snapshots")

    raise RuntimeError("No graph snapshot found. Start the simulator/live graph writer and pass its snapshot_id to Multiagents.")


def get_current_scenario() -> Tuple[List[App], List[Slice], List[Node]]:
    apps, slices, nodes = get_cached_scenario()
    if apps is not None and slices is not None and nodes is not None:
        return apps, slices, nodes
    return get_initial_scenario()


def get_current_optimizer_scenario() -> Tuple[List[App], List[Slice], List[Node]]:
    """Prefer the current run cache over the global latest graph snapshot."""
    apps, slices, nodes = get_cached_scenario()
    if apps is not None and slices is not None and nodes is not None:
        return apps, slices, nodes

    try:
        apps, slices, nodes, _snapshot_id = _load_latest_graph_scenario_strict()
        cache_scenario(
            apps,
            slices,
            nodes,
            _build_mobility_snapshot_payload(apps),
            _build_policy_state_payload(apps),
            snapshot_id=_snapshot_id,
        )
        return apps, slices, nodes
    except Exception as exc:
        logger.warning(f"Failed to load optimizer scenario from latest graph snapshot: {exc}")

    return get_current_scenario()


def initialize_scenario(reset: bool = False, scenario_file: Optional[Union[str, Path]] = None) -> dict:
    """Initialize local scenario cache/UE tables and return summary.

    Args:
        reset: If True, force-generate default scenario only in local cache/UE tables.
               Graph snapshots are never created or modified here.
    """
    if scenario_file is not None:
        scenario_path = Path(scenario_file).expanduser().resolve()
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
        apps, slices, nodes = _create_scenario_from_yaml(scenario_path)
        mobility_payload = _build_mobility_snapshot_payload(apps)
        policy_payload = _build_policy_state_payload(apps)
        cache_scenario(apps, slices, nodes, mobility_payload, policy_payload)
        seeded = _seed_ue_contexts_from_apps(apps)
        sync_summary = sync_latest_flow_five_tuples_to_ue_context()
        print(f"[UEContext] seeded {seeded} UE records (from scenario file)")
        print(f"[UEContext] synced five-tuples for {sync_summary['ues']} UEs / {sync_summary['flows']} flows (from scenario file)")
    elif reset:
        # 关键步骤：强制生成默认场景并保存
        apps, slices, nodes = _create_default_scenario()
        mobility_payload = _build_mobility_snapshot_payload(apps)
        policy_payload = _build_policy_state_payload(apps)
        cache_scenario(apps, slices, nodes, mobility_payload, policy_payload)
        seeded = _seed_ue_contexts_from_apps(apps)
        sync_summary = sync_latest_flow_five_tuples_to_ue_context()
        print(f"[UEContext] seeded {seeded} UE records (from reset)")
        print(f"[UEContext] synced five-tuples for {sync_summary['ues']} UEs / {sync_summary['flows']} flows (from reset)")
    else:
        apps, slices, nodes = get_initial_scenario()

    an_count = sum(1 for n in nodes if getattr(n, "node_type", "") == "AN")
    cn_count = sum(1 for n in nodes if getattr(n, "node_type", "") == "CN")
    min_flows = min((len(a.flows) for a in apps), default=0)
    max_flows = max((len(a.flows) for a in apps), default=0)

    return {
        "apps": len(apps),
        "slices": len(slices),
        "nodes": len(nodes),
        "an_nodes": an_count,
        "cn_nodes": cn_count,
        "flow_range_per_app": [min_flows, max_flows],
        "mode": "scenario-file" if scenario_file is not None else ("reset" if reset else "load-or-init"),
        "scenario_file": str(Path(scenario_file).expanduser().resolve()) if scenario_file is not None else "",
    }


def init_main() -> None:
    parser = argparse.ArgumentParser(description="Initialize scenario cache and DB snapshot")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Legacy local-cache reset only; does not write graph snapshots",
    )
    parser.add_argument(
        "--scenario-file",
        help="Legacy local-cache YAML seed only; does not write graph snapshots",
    )
    parser.add_argument(
        "--rebuild-from-latest-graph",
        action="store_true",
        help="Rebuild cache and UE-related tables from the latest graph snapshot",
    )
    parser.add_argument(
        "--graph-snapshot-id",
        default="",
        help="Existing network graph snapshot id to read when rebuilding UE-related tables",
    )
    args = parser.parse_args()

    if args.graph_snapshot_id:
        summary = rebuild_ue_related_tables_from_graph_snapshot(args.graph_snapshot_id)
        print("Scenario rebuilt from graph snapshot:")
        print(summary)
        return

    if args.rebuild_from_latest_graph:
        summary = rebuild_ue_related_tables_from_latest_graph()
        print("Scenario rebuilt from latest graph:")
        print(summary)
        return

    summary = initialize_scenario(reset=args.reset, scenario_file=args.scenario_file)
