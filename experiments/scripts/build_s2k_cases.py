from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.contracts import ExperimentCase
from training.artifacts import write_jsonl


@dataclass(frozen=True)
class Family:
    code: str
    templates: int
    profiles: tuple[str, ...]
    feasibility: str
    resolution: str
    reason_code: str
    operation_class: str
    main_domains: tuple[str, ...]
    sba_evidence: tuple[str, ...]
    ppa_plan: str
    evidence_contract: tuple[str, ...]
    acceptance_checks: tuple[str, ...]


FLOWS = (
    ("imsi-208930000000001", "remote-drive-video-1", "remote-drive", "embb", "01000001"),
    ("imsi-208930000000006", "factory-robot-video-1", "factory-robot", "embb", "01000001"),
    ("imsi-208930000000006", "iot-sensor-telemetry-1", "iot-sensor", "mmtc", "03000001"),
    ("imsi-208930000000005", "ar-gaming-control-1", "ar-gaming", "urllc", "02000001"),
    ("imsi-208930000000008", "telemedicine-video-1", "telemedicine", "urllc", "02000001"),
    ("imsi-208930000000009", "video-4k-stream-1", "video-4k", "embb", "01000001"),
    ("imsi-208930000000010", "cloud-render-stream-1", "cloud-render", "urllc", "02000001"),
)

SCENES = (
    {"id": "remote_driving_edge", "site": "城市道路边缘站", "condition": "高速移动与低时延控制", "flow_index": 0},
    {"id": "smart_factory", "site": "离散制造车间", "condition": "机器人协同与产线高负载", "flow_index": 1},
    {"id": "industrial_iot", "site": "工业园区传感网", "condition": "海量遥测与低优先级竞争", "flow_index": 2},
    {"id": "stadium_xr", "site": "体育场馆", "condition": "高密度 XR 互动", "flow_index": 3},
    {"id": "smart_hospital", "site": "医院急诊区", "condition": "远程诊疗与可靠性保护", "flow_index": 4},
    {"id": "live_video_venue", "site": "大型活动现场", "condition": "4K 视频上行拥塞", "flow_index": 5},
    {"id": "edge_cloud", "site": "园区边缘云", "condition": "云渲染峰值与双业务并行", "flow_index": 6},
    {"id": "emergency_response", "site": "灾害救援现场", "condition": "应急指挥与资源抢占", "flow_index": 0},
    {"id": "warehouse_logistics", "site": "智能仓储", "condition": "移动机器人与视频巡检", "flow_index": 1},
    {"id": "smart_city_sensing", "site": "城市感知网络", "condition": "广域遥测与背景流竞争", "flow_index": 2},
)

PROFILES: dict[str, dict[str, Any]] = {
    "qos_normal": {
        "subscription": ["01000001", "02000001", "03000001"],
        "pdu_pool": ["embb/01000001", "urllc/02000001", "mmtc/03000001"],
        "capability_inventory": ["qos", "existing_pdu_ursp", "am_rfsp"],
        "load_class": "normal",
    },
    "qos_congested": {
        "subscription": ["01000001", "02000001", "03000001"],
        "pdu_pool": ["embb/01000001", "urllc/02000001", "mmtc/03000001"],
        "capability_inventory": ["qos", "existing_pdu_ursp", "am_rfsp"],
        "load_class": "congested",
    },
    "qos_contention": {
        "subscription": ["01000001", "02000001", "03000001"],
        "pdu_pool": ["embb/01000001", "urllc/02000001", "mmtc/03000001"],
        "capability_inventory": ["qos", "existing_pdu_ursp", "am_rfsp"],
        "load_class": "contention",
    },
    "dual_pdu_ursp": {
        "subscription": ["01000001", "02000001", "03000001"],
        "pdu_pool": ["embb/01000001", "urllc/02000001"],
        "capability_inventory": ["qos", "existing_pdu_ursp"],
        "load_class": "normal",
    },
    "am_rfsp_only": {
        "subscription": ["01000001", "02000001", "03000001"],
        "pdu_pool": ["embb/01000001", "urllc/02000001", "mmtc/03000001"],
        "capability_inventory": ["am_rfsp"],
        "load_class": "normal",
    },
    "am_ursp_authorized": {
        "subscription": ["01000001", "02000001", "03000001"],
        "pdu_pool": ["embb/01000001", "urllc/02000001"],
        "capability_inventory": ["am_rfsp", "existing_pdu_ursp"],
        "load_class": "normal",
    },
    "multi_supi_staged": {
        "subscription": ["01000001", "02000001", "03000001"],
        "pdu_pool": ["embb/01000001", "urllc/02000001", "mmtc/03000001"],
        "capability_inventory": ["qos", "existing_pdu_ursp", "am_rfsp"],
        "load_class": "contention",
    },
    "subscription_denied": {
        "subscription": ["01000001"],
        "pdu_pool": ["embb/01000001"],
        "capability_inventory": ["qos", "existing_pdu_ursp"],
        "load_class": "normal",
    },
    "pdu_pool_gap": {
        "subscription": ["01000001", "02000001"],
        "pdu_pool": ["embb/01000001"],
        "capability_inventory": ["qos", "existing_pdu_ursp"],
        "load_class": "normal",
    },
    "capability_gap": {
        "subscription": ["01000001", "02000001", "03000001"],
        "pdu_pool": ["embb/01000001", "urllc/02000001", "mmtc/03000001"],
        "capability_inventory": ["qos", "existing_pdu_ursp"],
        "load_class": "normal",
    },
    "safety_conflict": {
        "subscription": ["01000001", "02000001", "03000001"],
        "pdu_pool": ["embb/01000001", "urllc/02000001", "mmtc/03000001"],
        "capability_inventory": ["qos", "existing_pdu_ursp", "am_rfsp"],
        "load_class": "contention",
    },
}

FAMILIES = (
    Family("F1", 18, ("qos_normal", "qos_congested", "qos_contention"), "feasible", "executed", "objective_achieved", "qos_same_pdu", ("qos",), ("flow_catalog", "qos_policy_binding"), "qos_only", ("snapshot_id", "flow_catalog", "qos_binding"), ("receipt_success", "qos_changed", "route_unchanged")),
    Family("F2", 16, ("dual_pdu_ursp",), "feasible", "executed", "objective_achieved", "ursp_existing_pdu", ("qos",), ("flow_catalog", "subscription", "pdu_session_pool"), "existing_pdu_only", ("snapshot_id", "flow_catalog", "pdu_session_pool"), ("receipt_success", "target_flow_route_changed")),
    Family("F3", 10, ("am_rfsp_only", "am_ursp_authorized"), "feasible", "executed", "objective_achieved", "am_ursp", ("mobility",), ("am_policy_association", "subscription", "pdu_session_pool"), "am_or_existing_pdu", ("snapshot_id", "am_policy_association"), ("receipt_success", "am_policy_version_advanced")),
    Family("F4", 8, ("multi_supi_staged",), "feasible", "executed", "objective_achieved", "multi_supi_staged", ("qos",), ("flow_catalog", "subscription", "resource_snapshot"), "staged_existing_pdu", ("snapshot_id", "flow_catalog", "resource_snapshot"), ("receipt_success", "stage_order_preserved")),
    Family("F5", 8, ("qos_congested", "qos_contention"), "feasible", "executed", "objective_achieved", "safe_qos_execution", ("qos",), ("flow_catalog", "resource_snapshot"), "safe_qos_only", ("snapshot_id", "flow_catalog", "resource_snapshot"), ("receipt_success", "safety_budget_preserved")),
    Family("I1", 14, ("subscription_denied",), "infeasible", "infeasible", "blocked_by_subscription_entitlement", "subscription_entitlement_block", ("qos",), ("flow_catalog", "subscription"), "entitlement_partial_plan", ("snapshot_id", "flow_catalog", "subscription"), ("partial_plan", "no_action", "no_open_questions")),
    Family("I2", 10, ("pdu_pool_gap",), "infeasible", "infeasible", "dynamic_pdu_required", "pdu_pool_gap", ("qos",), ("flow_catalog", "subscription", "pdu_session_pool"), "dynamic_pdu_required", ("snapshot_id", "flow_catalog", "subscription", "pdu_session_pool"), ("partial_plan", "no_action", "no_open_questions")),
    Family("I3", 8, ("capability_gap",), "infeasible", "infeasible", "capability_unavailable", "capability_gap", ("mobility",), ("capability_inventory", "flow_catalog"), "capability_partial_plan", ("snapshot_id", "capability_inventory"), ("partial_plan", "no_action", "no_open_questions")),
    Family("I4", 8, ("safety_conflict",), "infeasible", "infeasible", "constraint_unsatisfied", "safety_constraint_conflict", ("qos",), ("flow_catalog", "subscription", "resource_snapshot"), "constraint_partial_plan", ("snapshot_id", "flow_catalog", "resource_snapshot"), ("partial_plan", "no_action", "no_open_questions")),
)


def _partition(template_ordinal: int) -> str:
    if template_ordinal <= 70:
        return "train"
    if template_ordinal <= 85:
        return "validation"
    return "test"


SNSSAI_DNN = {
    "01000001": "embb",
    "02000001": "urllc",
    "03000001": "mmtc",
}


def _alternate_snssai(current: str, template: int) -> str:
    choices = [snssai for snssai in SNSSAI_DNN if snssai != current]
    return choices[template % len(choices)]


def _target_snssai(
    family: Family, template: int, flow: tuple[str, str, str, str, str]
) -> str | None:
    current = flow[4]
    if family.code in {"F2", "I1", "I2"}:
        return _alternate_snssai(current, template)
    if family.code == "F3" and template % 2 == 0:
        return _alternate_snssai(current, template)
    return None


def _initial_state(
    family: Family, profile: str, flow: tuple[str, str, str, str, str], target: str | None
) -> dict[str, Any]:
    source = PROFILES[profile]
    supi, _, _, dnn, current = flow
    subscription = list(source["subscription"])
    pdu_pool = list(source["pdu_pool"])
    current_pdu = f"{dnn}/{current}"
    if current_pdu not in pdu_pool:
        pdu_pool.append(current_pdu)
    if family.code == "I1":
        subscription = [current]
    elif family.code == "I2":
        subscription = [current, target]
        pdu_pool = [item for item in pdu_pool if not item.endswith(target)]
    elif target is not None:
        target_pdu = f"{SNSSAI_DNN[target]}/{target}"
        if target_pdu not in pdu_pool:
            pdu_pool.append(target_pdu)
    return {
        "subscription": subscription,
        "pdu_pool": pdu_pool,
        "capability_inventory": list(source["capability_inventory"]),
        "load_class": source["load_class"],
        "bound_supi": supi,
    }


def _request(
    family: Family,
    template: int,
    variant: int,
    flow: tuple[str, str, str, str, str],
    target: str | None,
) -> str:
    supi, flow_id, app_id, dnn, snssai = flow
    style = ("请", "需要", "请立即", "请优先", "请在不影响其他业务的前提下")[variant % 5]
    if family.code == "F1":
        metric = ("上行保障带宽", "下行保障带宽", "时延目标", "抖动目标", "业务优先级", "GBR")[template % 6]
        return f"{style}调整 {supi} 的 {flow_id}（{app_id}）{metric}，保持当前 {dnn}/{snssai} PDU 与路由不变。"
    if family.code == "F2":
        return f"{style}为 {supi} 的 {flow_id} 配置到已存在 {target} PDU 的 URSP 选路，不创建新 PDU。"
    if family.code == "F3":
        if target is None:
            return f"{style}仅调整 {supi} 的 RFSP，保持 {flow_id} 当前 PDU 路由不变。"
        return f"{style}调整 {supi} 的 RFSP，并把 {flow_id} 选路到已存在的授权 {target} PDU。"
    if family.code == "F4":
        peer = FLOWS[(template + variant + 1) % len(FLOWS)][0]
        return f"{style}先保护 {supi} 的 {flow_id}，再处理 {peer} 的低优先级流；必须逐 SUPI 分阶段执行。"
    if family.code == "F5":
        return f"{style}在峰值负载下改善 {supi} 的 {flow_id}，不得降低受保护流的保障，也不得创建额外 URSP。"
    if family.code == "I1":
        return f"{style}将 {supi} 的 {flow_id} 迁移至 {target}，不修改用户订阅。"
    if family.code == "I2":
        return f"{style}将 {supi} 的 {flow_id} 选路至已授权但当前未建立的 {target} PDU，且不创建动态 PDU。"
    if family.code == "I3":
        capability = ("实际无线重选", "未支持的 AM 字段", "未部署的 URSP profile")[template % 3]
        return f"{style}为 {supi} 的 {flow_id} 执行 {capability}。"
    return f"{style}同时迁移 {supi} 的 {flow_id} 并保持其当前路由、订阅和保护优先级全部不变。"


def _cases(families: Iterable[Family], run_id: str) -> list[ExperimentCase]:
    cases: list[ExperimentCase] = []
    template_ordinal = 0
    for family in families:
        for template in range(1, family.templates + 1):
            template_ordinal += 1
            for variant in range(1, 11):
                scene = SCENES[variant - 1]
                flow = FLOWS[scene["flow_index"]]
                supi, flow_id, app_id, dnn, snssai = flow
                profile = family.profiles[(template + variant - 2) % len(family.profiles)]
                target = _target_snssai(family, template, flow)
                initial_state = _initial_state(family, profile, flow, target)
                catalog_id = f"S2K-{family.code}-{template:03d}-v{variant:02d}"
                cases.append(
                    ExperimentCase(
                        case_id=catalog_id,
                        episode_id=f"{run_id}-{catalog_id.lower()}",
                        method="multi_agent_control",
                        user_input=(
                            f"在{scene['site']}的{scene['condition']}场景中，" + _request(family, template, variant, flow, target)
                        ),
                        metadata={
                            "catalog_version": "s2k-v3",
                            "catalog_id": catalog_id,
                            "partition": _partition(template_ordinal),
                            "template_ordinal": template_ordinal,
                            "variant": variant,
                            "scenario_profile": profile,
                            "scenario_initial_state": initial_state,
                            "scenario_context": scene,
                            "flow_binding": {"supi": supi, "flow_id": flow_id, "app_id": app_id, "dnn": dnn, "snssai": snssai},
                            "requested_route": {"target_snssai": target, "requires_existing_pdu": family.code in {"F2", "F3"}},
                            "operation_class": family.operation_class,
                            "feasibility": family.feasibility,
                            "expected_terminal": {"resolution": family.resolution, "reason_code": family.reason_code},
                            "expected_agent_path": {"main_domains": list(family.main_domains), "sba_evidence": list(family.sba_evidence), "ppa_plan": family.ppa_plan},
                            "evidence_contract": list(family.evidence_contract),
                            "acceptance_checks": list(family.acceptance_checks),
                            "sft_disposition": "pending_execution",
                        },
                    )
                )
    return cases


def build_s2k_cases(*, run_id: str) -> list[ExperimentCase]:
    cases = _cases(FAMILIES, run_id)
    if len(cases) != 1000:
        raise ValueError(f"expected 1000 S2K cases, found {len(cases)}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = build_s2k_cases(run_id=args.run_id)
    write_jsonl(args.output, cases)
    print(f"Wrote {len(cases)} S2K catalog cases to {args.output}")


if __name__ == "__main__":
    main()


__all__ = ["FAMILIES", "PROFILES", "build_s2k_cases"]

