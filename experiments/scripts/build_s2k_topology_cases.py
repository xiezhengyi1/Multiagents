from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments.contracts import ExperimentCase
from training.artifacts import write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_ROOT = (
    PROJECT_ROOT.parent / "oai-free5gc-testbed" / "scenarios" / "experiments"
)
TOPOLOGY_KEYS = ("networks", "slices", "sites", "core", "ran", "ues", "mobility")
TRAFFIC_KEYS = ("services", "links", "flows")
TRAIN_SCENARIOS = frozenset((*range(1, 13), 15, 16))
VALIDATION_SCENARIOS = frozenset((13, 17, 19))
TEST_SCENARIOS = frozenset((14, 18, 20))
EXTERNAL_ROUTES = (
    {"dnn": "embb", "snssai": "01000001"},
    {"dnn": "urllc", "snssai": "02000001"},
    {"dnn": "mmtc", "snssai": "03000001"},
    {"dnn": "mec", "snssai": "01010203"},
    {"dnn": "internet", "snssai": "01010203"},
)
SCENE_LABELS = {
    1: "乡村道路低时延控制",
    2: "工厂单元机器人控制",
    3: "智能工厂混合业务",
    4: "园区物联网遥测",
    5: "城区海量物联网",
    6: "体育场高密度宽带",
    7: "医院园区混合业务",
    8: "应急走廊控制",
    9: "仓储机器人协同",
    10: "大学园区混合业务",
    11: "高速公路移动控制",
    12: "城市移动混合业务",
    13: "郊区均衡负载",
    14: "密集视频热点",
    15: "小型边云协同",
    16: "园区边云协同",
    17: "密集边缘推理",
    18: "区域云端推理",
    19: "退化云回传",
    20: "灾害场景有损回传",
}
SERVICE_LABELS = {
    "remote-drive": "远程驾驶的视频画面",
    "factory-robot": "工厂机器人的控制业务",
    "ar-gaming": "AR 交互业务",
    "iot-sensor": "传感器遥测业务",
    "telemedicine": "远程诊疗视频",
    "video-4k": "4K 视频业务",
    "cloud-render": "云端渲染交互业务",
    "inference": "边缘智能推理业务",
    "web-browse": "网页浏览业务",
}
QOS_REQUESTS = {
    "上行保障带宽": "提升上行传输保障",
    "下行保障带宽": "提升下行传输保障",
    "时延目标": "降低端到端时延",
    "抖动目标": "减少时延抖动",
    "丢包率目标": "降低丢包",
    "业务优先级": "提高业务优先级",
}


@dataclass(frozen=True)
class ScenarioRecord:
    number: int
    path: Path
    payload: dict[str, Any]
    config_sha256: str
    topology_sha256: str
    traffic_sha256: str
    partition: str


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _partition(number: int) -> str:
    if number in TRAIN_SCENARIOS:
        return "train"
    if number in VALIDATION_SCENARIOS:
        return "validation"
    if number in TEST_SCENARIOS:
        return "test"
    raise ValueError(f"scenario {number} has no dataset partition")


def load_scenarios(scenario_root: Path = DEFAULT_SCENARIO_ROOT) -> list[ScenarioRecord]:
    paths = sorted(scenario_root.glob("s2k-*.yaml"))
    if len(paths) != 20:
        raise ValueError(f"expected 20 topology scenarios, found {len(paths)}")
    records = []
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        number = int(payload["scenario_id"].split("-")[1])
        records.append(
            ScenarioRecord(
                number=number,
                path=path.resolve(strict=True),
                payload=payload,
                config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                topology_sha256=_digest({key: payload[key] for key in TOPOLOGY_KEYS}),
                traffic_sha256=_digest({key: payload[key] for key in TRAFFIC_KEYS}),
                partition=_partition(number),
            )
        )
    return records


def _snssai(slices: dict[str, dict[str, Any]], slice_id: str) -> str:
    item = slices[slice_id]
    return f"{item['sst']:02d}{item['sd']}"


def _inventory(record: ScenarioRecord) -> dict[str, Any]:
    payload = record.payload
    return {
        "ues": {item["id"]: item for item in payload["ues"]},
        "slices": {item["id"]: item for item in payload["slices"]},
        "services": {item["id"]: item for item in payload["services"]},
        "flows": payload["flows"],
    }


def _flow_binding(
    flow: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    ue = inventory["ues"][flow["ue_id"]]
    session = next(item for item in ue["sessions"] if item["id"] == flow["session_id"])
    service = inventory["services"][flow["service_instance_id"]]
    return {
        "supi": ue["supi"],
        "ue_id": ue["id"],
        "serving_gnb": ue["serving_gnb"],
        "flow_id": flow["id"],
        "app_id": service["service_id"],
        "service_instance_id": service["id"],
        "service_site_id": service["site_id"],
        "protocol": flow["protocol"],
        "src_port": flow["src_port"],
        "dst_port": flow["dst_port"],
        "session_id": session["id"],
        "dnn": session["dnn"],
        "slice_id": session["slice_id"],
        "snssai": _snssai(inventory["slices"], session["slice_id"]),
    }


def _flow_description(binding: dict[str, Any]) -> str:
    return SERVICE_LABELS[binding["app_id"]]


def _session_pool(
    binding: dict[str, Any],
    inventory: dict[str, Any],
) -> list[dict[str, str]]:
    ue = inventory["ues"][binding["ue_id"]]
    return [
        {
            "session_id": session["id"],
            "dnn": session["dnn"],
            "slice_id": session["slice_id"],
            "snssai": _snssai(inventory["slices"], session["slice_id"]),
        }
        for session in ue["sessions"]
    ]


def _existing_target(
    binding: dict[str, Any],
    inventory: dict[str, Any],
    ordinal: int,
) -> dict[str, str]:
    alternatives = [
        item
        for item in _session_pool(binding, inventory)
        if (item["dnn"], item["snssai"]) != (binding["dnn"], binding["snssai"])
    ]
    return alternatives[ordinal % len(alternatives)]


def _unauthorized_target(
    binding: dict[str, Any],
    inventory: dict[str, Any],
    ordinal: int,
) -> dict[str, str]:
    authorized = {
        (item["dnn"], item["snssai"]) for item in _session_pool(binding, inventory)
    }
    declared = [
        {
            "dnn": session["dnn"],
            "snssai": _snssai(inventory["slices"], session["slice_id"]),
        }
        for ue in inventory["ues"].values()
        for session in ue["sessions"]
        if (session["dnn"], _snssai(inventory["slices"], session["slice_id"]))
        not in authorized
    ]
    candidates = declared + [
        item
        for item in EXTERNAL_ROUTES
        if (item["dnn"], item["snssai"]) not in authorized
    ]
    return candidates[ordinal % len(candidates)]


def _scenario_metadata(record: ScenarioRecord) -> dict[str, Any]:
    payload = record.payload
    return {
        "scenario_id": payload["scenario_id"],
        "scenario_label": SCENE_LABELS[record.number],
        "scenario_config_path": str(record.path),
        "scenario_config_relative_path": f"scenarios/experiments/{record.path.name}",
        "scenario_config_sha256": record.config_sha256,
        "network_topology_sha256": record.topology_sha256,
        "traffic_topology_sha256": record.traffic_sha256,
        "topology_counts": {
            "sites": len(payload["sites"]),
            "slices": len(payload["slices"]),
            "upfs": len(payload["core"]["upfs"]),
            "gnbs": len(payload["ran"]["gnbs"]),
            "ues": len(payload["ues"]),
            "services": len(payload["services"]),
            "links": len(payload["links"]),
            "flows": len(payload["flows"]),
            "mobility_events": len(payload["mobility"]),
        },
    }


def _initial_state(
    binding: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    pool = _session_pool(binding, inventory)
    return {
        "bound_supi": binding["supi"],
        "serving_gnb": binding["serving_gnb"],
        "subscription": pool,
        "pdu_pool": pool,
    }


def _case(
    *,
    record: ScenarioRecord,
    run_id: str,
    case_number: int,
    user_input: str,
    bindings: list[dict[str, Any]],
    operation_class: str,
    feasibility: str,
    reason_code: str,
    requested_route: dict[str, Any] | None,
    main_domains: list[str],
    sba_evidence: list[str],
    ppa_plan: str,
    evidence_contract: list[str],
    acceptance_checks: list[str],
    inventory: dict[str, Any],
) -> ExperimentCase:
    case_id = f"S2K-T{record.number:02d}-{case_number:02d}"
    metadata = {
        "catalog_version": "s2k-topology-v3-balanced",
        "catalog_id": case_id,
        "partition": record.partition,
        **_scenario_metadata(record),
        "case_class": feasibility,
        "correct_behavior": "execute" if feasibility == "feasible" else "refuse",
        "operation_class": operation_class,
        "flow_binding": bindings[0],
        "flow_bindings": bindings,
        "scenario_initial_state": _initial_state(bindings[0], inventory),
        "scenario_initial_states": [
            _initial_state(binding, inventory) for binding in bindings
        ],
        "requested_route": requested_route,
        "expected_terminal": {
            "outcome": "success",
            "resolution": "executed" if feasibility == "feasible" else "infeasible",
            "reason_code": reason_code,
        },
        "expected_agent_path": {
            "main_domains": main_domains,
            "sba_evidence": sba_evidence,
            "ppa_plan": ppa_plan,
        },
        "evidence_contract": evidence_contract,
        "acceptance_checks": acceptance_checks,
        "sft_disposition": "pending_execution",
    }
    return ExperimentCase(
        case_id=case_id,
        episode_id=f"{run_id}-{case_id.lower()}",
        method="multi_agent_control",
        user_input=user_input,
        metadata=metadata,
    )


def _scenario_cases(record: ScenarioRecord, run_id: str) -> list[ExperimentCase]:
    inventory = _inventory(record)
    flows = inventory["flows"]
    bindings = [_flow_binding(flow, inventory) for flow in flows]
    multi_route_bindings = [
        binding for binding in bindings if len(_session_pool(binding, inventory)) > 1
    ]
    cases = []

    metrics = (
        "上行保障带宽",
        "下行保障带宽",
        "时延目标",
        "抖动目标",
        "丢包率目标",
        "业务优先级",
    )
    for index in range(1, 19):
        binding = bindings[(index - 1) % len(bindings)]
        metric = metrics[(index - 1) % len(metrics)]
        cases.append(
            _case(
                record=record,
                run_id=run_id,
                case_number=index,
                user_input=(
                    f"请为用户 SUPI {binding['supi']} 的{_flow_description(binding)}"
                    f"{QOS_REQUESTS[metric]}约 {5 + index}% ，"
                    "保持它现在使用的业务连接不变。"
                ),
                bindings=[binding],
                operation_class="qos_same_pdu",
                feasibility="feasible",
                reason_code="objective_achieved",
                requested_route=None,
                main_domains=["qos"],
                sba_evidence=["flow_catalog", "qos_policy_binding"],
                ppa_plan="qos_only",
                evidence_contract=["snapshot_id", "flow_catalog", "qos_binding"],
                acceptance_checks=["receipt_success", "qos_changed", "route_unchanged"],
                inventory=inventory,
            )
        )

    for index in range(19, 29):
        binding = multi_route_bindings[(index - 19) % len(multi_route_bindings)]
        target = _existing_target(binding, inventory, index)
        cases.append(
            _case(
                record=record,
                run_id=run_id,
                case_number=index,
                user_input=(
                    f"用户 SUPI {binding['supi']} 的{_flow_description(binding)}需要走已经可用的"
                    f"低时延业务连接，并将它设为第 {index - 18} 优先选择，请不要新建连接。"
                ),
                bindings=[binding],
                operation_class="ursp_existing_pdu",
                feasibility="feasible",
                reason_code="objective_achieved",
                requested_route={**target, "requires_existing_pdu": True},
                main_domains=["qos"],
                sba_evidence=["flow_catalog", "subscription", "pdu_session_pool"],
                ppa_plan="existing_pdu_ursp",
                evidence_contract=[
                    "snapshot_id",
                    "flow_catalog",
                    "subscription",
                    "pdu_session_pool",
                    "ue_policy_association",
                ],
                acceptance_checks=[
                    "receipt_success",
                    "target_flow_route_changed",
                    "no_dynamic_pdu",
                ],
                inventory=inventory,
            )
        )

    for index in range(29, 39):
        binding = bindings[(index - 29) % len(bindings)]
        cases.append(
            _case(
                record=record,
                run_id=run_id,
                case_number=index,
                user_input=(
                    f"请只调整用户 SUPI {binding['supi']} 的移动性优先级和驻留偏好，"
                    f"调整幅度为第 {index - 28} 档，不要改变其{_flow_description(binding)}正在使用的业务连接。"
                ),
                bindings=[binding],
                operation_class="am_rfsp_only",
                feasibility="feasible",
                reason_code="objective_achieved",
                requested_route=None,
                main_domains=["mobility"],
                sba_evidence=["am_policy_association", "subscription"],
                ppa_plan="am_only",
                evidence_contract=[
                    "snapshot_id",
                    "am_policy_association",
                    "subscription",
                ],
                acceptance_checks=[
                    "receipt_success",
                    "am_policy_version_advanced",
                    "route_unchanged",
                ],
                inventory=inventory,
            )
        )

    multi_supi_priorities = (
        "稳定",
        "连续",
        "低时延",
        "低抖动",
        "上行保障",
        "下行保障",
        "低丢包",
        "可用性",
        "控制响应",
        "视频质量",
    )
    for index in range(39, 49):
        primary = bindings[(index - 39) % len(bindings)]
        peer = next(item for item in bindings if item["supi"] != primary["supi"])
        cases.append(
            _case(
                record=record,
                run_id=run_id,
                case_number=index,
                user_input=(
                    f"先确保用户 SUPI {primary['supi']} 的{_flow_description(primary)}"
                    f"{multi_supi_priorities[index - 39]}，"
                    f"再处理用户 SUPI {peer['supi']} 的{_flow_description(peer)}，请分两步完成。"
                ),
                bindings=[primary, peer],
                operation_class="multi_supi_staged",
                feasibility="feasible",
                reason_code="objective_achieved",
                requested_route=None,
                main_domains=["qos"],
                sba_evidence=["flow_catalog", "subscription", "resource_snapshot"],
                ppa_plan="staged_qos",
                evidence_contract=["snapshot_id", "flow_catalog", "resource_snapshot"],
                acceptance_checks=["receipt_success", "stage_order_preserved"],
                inventory=inventory,
            )
        )

    safe_goals = (
        "适度改善体验",
        "改善响应速度",
        "改善传输稳定性",
        "改善画面连续性",
        "改善控制及时性",
        "改善上行表现",
        "改善下行表现",
        "改善时延表现",
        "改善抖动表现",
        "改善丢包表现",
    )
    for index in range(49, 59):
        binding = bindings[(index - 49) % len(bindings)]
        cases.append(
            _case(
                record=record,
                run_id=run_id,
                case_number=index,
                user_input=(
                    f"当前负载较高，请{safe_goals[index - 49]}用户 SUPI {binding['supi']} 的"
                    f"{_flow_description(binding)}；不能牺牲更重要业务的保障，也不要改变它正在使用的业务连接。"
                ),
                bindings=[binding],
                operation_class="safe_qos_execution",
                feasibility="feasible",
                reason_code="objective_achieved",
                requested_route=None,
                main_domains=["qos"],
                sba_evidence=["flow_catalog", "resource_snapshot"],
                ppa_plan="safe_qos_only",
                evidence_contract=["snapshot_id", "flow_catalog", "resource_snapshot"],
                acceptance_checks=[
                    "receipt_success",
                    "safety_budget_preserved",
                    "route_unchanged",
                ],
                inventory=inventory,
            )
        )

    migration_goals = (
        "降低时延",
        "提高上行吞吐",
        "提高下行吞吐",
        "减少抖动",
        "降低丢包",
        "保护控制稳定性",
        "释放拥塞链路",
        "提高业务优先级",
        "保障边缘处理",
        "避开退化回传",
    )
    for index in range(59, 69):
        binding = bindings[(index - 59) % len(bindings)]
        target = _unauthorized_target(binding, inventory, index)
        goal = migration_goals[index - 59]
        cases.append(
            _case(
                record=record,
                run_id=run_id,
                case_number=index,
                user_input=(
                    f"用户 SUPI {binding['supi']} 的{_flow_description(binding)}想要{goal}，"
                    "请切换到更合适的业务连接，但不要变更该用户的订阅。"
                ),
                bindings=[binding],
                operation_class="subscription_entitlement_block",
                feasibility="infeasible",
                reason_code="blocked_by_subscription_entitlement",
                requested_route={**target, "requires_existing_pdu": False},
                main_domains=["qos"],
                sba_evidence=["flow_catalog", "subscription"],
                ppa_plan="entitlement_partial_plan",
                evidence_contract=["snapshot_id", "flow_catalog", "subscription"],
                acceptance_checks=["partial_plan", "no_action", "no_open_questions"],
                inventory=inventory,
            )
        )

    conflict_goals = (
        "降低时延",
        "提高吞吐",
        "减少抖动",
        "降低丢包",
        "提升优先级",
        "稳定控制面",
        "避开拥塞",
        "切换到边缘",
        "切换到云端",
        "重排业务路径",
    )
    for index in range(69, 79):
        binding = multi_route_bindings[(index - 69) % len(multi_route_bindings)]
        target = _existing_target(binding, inventory, index)
        goal = conflict_goals[index - 69]
        cases.append(
            _case(
                record=record,
                run_id=run_id,
                case_number=index,
                user_input=(
                    f"为了{goal}，请为用户 SUPI {binding['supi']} 的{_flow_description(binding)}"
                    "换到更合适的业务连接，同时又必须完全保留现在的业务连接。"
                ),
                bindings=[binding],
                operation_class="hard_constraint_conflict",
                feasibility="infeasible",
                reason_code="constraint_unsatisfied",
                requested_route={**target, "requires_existing_pdu": True},
                main_domains=["qos"],
                sba_evidence=["flow_catalog", "subscription"],
                ppa_plan="constraint_partial_plan",
                evidence_contract=[
                    "snapshot_id",
                    "flow_catalog",
                    "subscription",
                    "user_constraints",
                ],
                acceptance_checks=["partial_plan", "no_action", "no_open_questions"],
                inventory=inventory,
            )
        )
    return cases


def build_s2k_topology_cases(
    *,
    run_id: str,
    scenario_root: Path = DEFAULT_SCENARIO_ROOT,
) -> list[ExperimentCase]:
    cases = [
        case
        for record in load_scenarios(scenario_root)
        for case in _scenario_cases(record, run_id)
    ]
    if len(cases) != 1560:
        raise ValueError(f"expected 1560 topology-bound cases, found {len(cases)}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scenario-root", type=Path, default=DEFAULT_SCENARIO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = build_s2k_topology_cases(
        run_id=args.run_id,
        scenario_root=args.scenario_root,
    )
    write_jsonl(args.output, cases)
    print(f"Wrote {len(cases)} topology-bound S2K cases to {args.output}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_SCENARIO_ROOT",
    "ScenarioRecord",
    "build_s2k_topology_cases",
    "load_scenarios",
]
