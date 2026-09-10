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


def _request(family: Family, template: int, variant: int, flow: tuple[str, str, str, str, str]) -> str:
    supi, flow_id, app_id, dnn, snssai = flow
    target = ("02000001", "03000001", "01000001")[template % 3]
    style = ("请", "需要", "请立即", "请优先", "请在不影响其他业务的前提下")[variant % 5]
    if family.code == "F1":
        metric = ("上行保障带宽", "下行保障带宽", "时延目标", "抖动目标", "业务优先级", "GBR")[template % 6]
        return f"{style}调整 {supi} 的 {flow_id}（{app_id}）{metric}，保持当前 {dnn}/{snssai} PDU 与路由不变。"
    if family.code == "F2":
        return f"{style}为 {supi} 的 {flow_id} 配置到已存在 {target} PDU 的 URSP 选路，不创建新 PDU。"
    if family.code == "F3":
        if template % 2:
            return f"{style}仅调整 {supi} 的 RFSP，保持 {flow_id} 当前 PDU 路由不变。"
        return f"{style}调整 {supi} 的 RFSP，并把 {flow_id} 选路到已存在的授权 PDU。"
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
                flow = FLOWS[(template + variant - 2) % len(FLOWS)]
                supi, flow_id, app_id, dnn, snssai = flow
                profile = family.profiles[(template + variant - 2) % len(family.profiles)]
                catalog_id = f"S2K-{family.code}-{template:03d}-v{variant:02d}"
                cases.append(
                    ExperimentCase(
                        case_id=catalog_id,
                        episode_id=f"{run_id}-{catalog_id.lower()}",
                        method="multi_agent_control",
                        user_input=_request(family, template, variant, flow),
                        metadata={
                            "catalog_version": "s2k-v1",
                            "catalog_id": catalog_id,
                            "partition": _partition(template_ordinal),
                            "template_ordinal": template_ordinal,
                            "variant": variant,
                            "scenario_profile": profile,
                            "scenario_initial_state": PROFILES[profile],
                            "flow_binding": {"supi": supi, "flow_id": flow_id, "app_id": app_id, "dnn": dnn, "snssai": snssai},
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

