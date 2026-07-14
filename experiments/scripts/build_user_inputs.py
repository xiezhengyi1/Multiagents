from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from experiments.paths import default_catalog_input_path, load_scenario_registry, scoped_catalog_input_path
from experiments.scripts.common import CONFIG_ROOT, PROJECT_ROOT, TASK_ROOT, load_yaml_mapping, write_json


TASK_CATALOG_PATH = TASK_ROOT / "task_catalog.json"
MATRIX_PATH = CONFIG_ROOT / "experiment_matrix.json"
CATEGORY_ORDER = [
    "slice_migration",
    "qos_adjustment",
    "resource_conflict",
    "multi_object_ambiguity",
    "cross_domain_coordination",
]
EXPECTED_CATEGORY_COUNTS = {
    "slice_migration": 7,
    "qos_adjustment": 7,
    "resource_conflict": 7,
    "multi_object_ambiguity": 7,
    "cross_domain_coordination": 8,
}
EXPECTED_TOTAL_COUNT = sum(EXPECTED_CATEGORY_COUNTS.values())


def _build_task_catalog() -> Dict[str, Any]:
    tasks: List[Dict[str, Any]] = [
        {
            "task_id": "T001",
            "category": "slice_migration",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "把 imsi-208930000000001 的 Remote_Drive_video_1 迁移到更低时延的切片，优先保证控制稳定性。",
            "expected_objects": {
                "supi": "imsi-208930000000001",
                "app": "Remote_Drive",
                "flow": "Remote_Drive_video_1",
            },
            "expected_direction": "迁移到低时延切片，保持高优先级保障",
            "success_criteria": "对象绑定正确，切片选择合理，执行后时延下降或满足目标",
        },
        {
            "task_id": "T002",
            "category": "slice_migration",
            "scenario_ids": ["S3", "S3P"],
            "user_input": "将 imsi-208930000000009 的 Drone_Control_video_1 调到更适合 URLLC 的切片，不要影响其当前业务连续性。",
            "expected_objects": {
                "supi": "imsi-208930000000009",
                "app": "Drone_Control",
                "flow": "Drone_Control_video_1",
            },
            "expected_direction": "迁移到更低时延/更高可靠切片",
            "success_criteria": "迁移目标与 flow 匹配，无错误切到大带宽普通切片",
        },
        {
            "task_id": "T003",
            "category": "slice_migration",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "把 Telemedicine_video_1 迁到更稳的医疗业务切片，重点看上行保障。",
            "expected_objects": {"app": "Telemedicine", "flow": "Telemedicine_video_1"},
            "expected_direction": "迁移到高可靠切片并提升上行保障",
            "success_criteria": "flow 识别正确，切片选择与医疗类业务匹配",
        },
        {
            "task_id": "T004",
            "category": "slice_migration",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "imsi-208930000000006 的 Factory_Robot_video_1 需要切换到工业控制优先的切片，避免与普通视频业务混跑。",
            "expected_objects": {
                "supi": "imsi-208930000000006",
                "app": "Factory_Robot",
                "flow": "Factory_Robot_video_1",
            },
            "expected_direction": "迁移到工业控制优先切片",
            "success_criteria": "切片隔离符合工业控制优先目标",
        },
        {
            "task_id": "T005",
            "category": "slice_migration",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "帮我把 AR_Gaming_video_2 从当前切片迁出去，换到更高吞吐且时延可控的切片。",
            "expected_objects": {"app": "AR_Gaming", "flow": "AR_Gaming_video_2"},
            "expected_direction": "迁移到高吞吐、低时延可接受切片",
            "success_criteria": "识别视频流而不是控制流，策略方向正确",
        },
        {
            "task_id": "T006",
            "category": "qos_adjustment",
            "scenario_ids": ["S1", "S2", "S3", "S1P", "S2P", "S3P"],
            "user_input": "提高 imsi-208930000000008 的 Telemedicine_video_1 下行带宽保障，并把目标时延压到更低。",
            "expected_objects": {
                "supi": "imsi-208930000000008",
                "app": "Telemedicine",
                "flow": "Telemedicine_video_1",
            },
            "expected_direction": "提高下行带宽保障并降低时延",
            "success_criteria": "QoS 字段完整，带宽和时延方向正确",
        },
        {
            "task_id": "T007",
            "category": "qos_adjustment",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "请优化 Cloud_Render_stream_1，只调整 SM policy，优先提高吞吐并限制抖动。",
            "expected_objects": {"app": "Cloud_Render", "flow": "Cloud_Render_stream_1"},
            "expected_direction": "提高吞吐、降低抖动，仅做 QoS 调整",
            "success_criteria": "requested_domains 仅含 qos，AM policy 不被误改",
        },
        {
            "task_id": "T008",
            "category": "qos_adjustment",
            "scenario_ids": ["S1", "S2", "S3", "S1P", "S2P", "S3P"],
            "user_input": "把 imsi-208930000000005 的 AR_Gaming_control_1 带宽再抬高一些，延迟尽量别超过 10ms。",
            "expected_objects": {
                "supi": "imsi-208930000000005",
                "app": "AR_Gaming",
                "flow": "AR_Gaming_control_1",
            },
            "expected_direction": "提升带宽、控制时延上限",
            "success_criteria": "flow 与 QoS 目标匹配，无对象串绑",
        },
        {
            "task_id": "T009",
            "category": "qos_adjustment",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "4K_Video_stream_1 的下行吞吐不够，帮我重算 QoS，但不要动 mobility。",
            "expected_objects": {"app": "4K_Video", "flow": "4K_Video_stream_1"},
            "expected_direction": "提高下行吞吐，仅调整 QoS",
            "success_criteria": "只生成 SM policy 相关策略",
        },
        {
            "task_id": "T010",
            "category": "qos_adjustment",
            "scenario_ids": ["S3", "S3P"],
            "user_input": "请针对 Drone_Control_video_1 提高上行保障，避免控制信令丢包。",
            "expected_objects": {"app": "Drone_Control", "flow": "Drone_Control_video_1"},
            "expected_direction": "提高上行带宽保障并降低丢包风险",
            "success_criteria": "QoS 优化方向正确，关键字段不缺失",
        },
        {
            "task_id": "T011",
            "category": "resource_conflict",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "资源紧张时优先保障 Factory_Robot_video_1，必要时压低 Web_Browse_session_1 的资源占用。",
            "expected_objects": {
                "primary_flow": "Factory_Robot_video_1",
                "secondary_flow": "Web_Browse_session_1",
            },
            "expected_direction": "优先工业控制流，牺牲低优先级业务",
            "success_criteria": "主次业务区分正确，资源重分配方向合理",
        },
        {
            "task_id": "T012",
            "category": "resource_conflict",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "如果切片资源不足，先保 Telemedicine_video_1 和 Remote_Drive_video_1，其余业务延后。",
            "expected_objects": {"flows": ["Telemedicine_video_1", "Remote_Drive_video_1"]},
            "expected_direction": "优先保障高优先级医疗和远程驾驶业务",
            "success_criteria": "多对象排序正确，优先级体现到策略上",
        },
        {
            "task_id": "T013",
            "category": "resource_conflict",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "先保 AR_Gaming_control_1 的时延，再看 Cloud_Render_stream_1 的吞吐，普通 IoT 最后处理。",
            "expected_objects": {
                "priority_flows": ["AR_Gaming_control_1", "Cloud_Render_stream_1"],
                "deprioritized_app": "IoT_Sensor",
            },
            "expected_direction": "按优先级分层处理资源",
            "success_criteria": "策略体现分层优先级，不把 IoT 提到高优先级",
        },
        {
            "task_id": "T014",
            "category": "resource_conflict",
            "scenario_ids": ["S3", "S3P"],
            "user_input": "上行带宽紧张，优先稳定 Drone_Control_video_1 和 Factory_Robot_video_1 的控制质量。",
            "expected_objects": {"flows": ["Drone_Control_video_1", "Factory_Robot_video_1"]},
            "expected_direction": "优先保障关键控制流的上行资源",
            "success_criteria": "上行资源调度方向正确",
        },
        {
            "task_id": "T015",
            "category": "resource_conflict",
            "scenario_ids": ["S3", "S3P"],
            "user_input": "如果必须降配，先从 4K_Video_stream_1 和 Web_Browse_session_2 开始，不要碰 Telemedicine_video_1。",
            "expected_objects": {
                "deprioritized_flows": ["4K_Video_stream_1", "Web_Browse_session_2"],
                "protected_flow": "Telemedicine_video_1",
            },
            "expected_direction": "优先降配低关键性业务",
            "success_criteria": "保护对象与降配对象不混淆",
        },
        {
            "task_id": "T016",
            "category": "multi_object_ambiguity",
            "scenario_ids": ["S1", "S2", "S3", "S1P", "S2P", "S3P"],
            "user_input": "把 telemedicine 那条视频流调稳一点，别动别的医疗业务。",
            "expected_objects": {"app": "Telemedicine", "flow": "Telemedicine_video_1"},
            "expected_direction": "定位到医疗控制流并提升稳定性",
            "success_criteria": "歧义对象解析正确，没有误伤其他医疗流",
        },
        {
            "task_id": "T017",
            "category": "multi_object_ambiguity",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "remote drive 那个视频流有点抖，帮我调一下，但别影响它的移动性策略。",
            "expected_objects": {"app": "Remote_Drive", "flow": "Remote_Drive_video_1"},
            "expected_direction": "只做 QoS 修正，不改 mobility",
            "success_criteria": "域判断正确，不把请求扩成联合控制",
        },
        {
            "task_id": "T018",
            "category": "multi_object_ambiguity",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "把 cloud render 的主业务切到更好的资源上，遥测那条先别动。",
            "expected_objects": {
                "app": "Cloud_Render",
                "target_flow": "Cloud_Render_stream_1",
                "excluded_flow": "Cloud_Render_telemetry_2",
            },
            "expected_direction": "只处理主业务流",
            "success_criteria": "主业务与遥测流区分正确",
        },
        {
            "task_id": "T019",
            "category": "multi_object_ambiguity",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "把那个机器人业务调优一下，优先视频控制那条，不是普通传感器。",
            "expected_objects": {"app": "Factory_Robot", "flow": "Factory_Robot_video_1"},
            "expected_direction": "定位到机器人控制流而不是 IoT_Sensor",
            "success_criteria": "避免对象误识别到 IoT 业务",
        },
        {
            "task_id": "T020",
            "category": "multi_object_ambiguity",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "请只调整 imsi-208930000000001 的 AM policy，重点检查 allowed NSSAI 和 RFSP，不要改 QoS。",
            "expected_objects": {"supi": "imsi-208930000000001"},
            "expected_direction": "只做 mobility/AM policy 调整",
            "success_criteria": "requested_domains 仅含 mobility，QoS 不被误改",
        },
        {
            "task_id": "T021",
            "category": "slice_migration",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "把 Remote_Drive_video_1 从当前切片迁到低时延资源上，但这次只做 QoS 侧迁移，不要改 allowed NSSAI 或 RFSP。",
            "expected_objects": {"app": "Remote_Drive", "flow": "Remote_Drive_video_1"},
            "expected_direction": "低时延 QoS 迁移，显式排除 AM policy 修改",
            "success_criteria": "能够区分 SM/QoS 切片重分配与 AM policy 同步，不误改 mobility",
        },
        {
            "task_id": "T022",
            "category": "slice_migration",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "把 4K_Video_stream_1 换到吞吐更高的切片，4K_Video_control_2 先别动；如果换片不可执行，就只提高下行保障。",
            "expected_objects": {
                "app": "4K_Video",
                "target_flow": "4K_Video_stream_1",
                "excluded_flow": "4K_Video_control_2",
            },
            "expected_direction": "优先切到高吞吐切片，失败时退化为 QoS 带宽调整",
            "success_criteria": "目标流与排除流区分正确，fallback 不改变控制流",
        },
        {
            "task_id": "T023",
            "category": "qos_adjustment",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "Cloud_Render_stream_1 先保证下行吞吐，抖动次要；Cloud_Render_telemetry_2 不动，也不要改 mobility。",
            "expected_objects": {
                "app": "Cloud_Render",
                "target_flow": "Cloud_Render_stream_1",
                "excluded_flow": "Cloud_Render_telemetry_2",
            },
            "expected_direction": "只对主渲染流做 QoS 调整，吞吐优先于抖动",
            "success_criteria": "不误伤 telemetry 流，不输出 AM policy",
        },
        {
            "task_id": "T024",
            "category": "qos_adjustment",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "Remote_Drive_video_1 的稳定性优先；如果低时延切片不可用，不要强行迁移，先降低 jitter 和 loss 并保持 AM policy 不变。",
            "expected_objects": {"app": "Remote_Drive", "flow": "Remote_Drive_video_1"},
            "expected_direction": "稳定性优先的 best-effort QoS 调整，保留 mobility",
            "success_criteria": "能把不可行迁移转成软 QoS 优化，并保持对象绑定",
        },
        {
            "task_id": "T025",
            "category": "resource_conflict",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "同一轮处理 Factory_Robot_video_1 和 IoT_Sensor_telemetry_1：保机器人控制稳定，必要时压低 IoT 遥测资源，不要动 UE allowed NSSAI。",
            "expected_objects": {
                "primary_flow": "Factory_Robot_video_1",
                "secondary_flow": "IoT_Sensor_telemetry_1",
            },
            "expected_direction": "保护工业控制流，降配同 UE 的低优先级遥测",
            "success_criteria": "能处理同 UE 多应用资源冲突，不把 IoT 当成目标保护流",
        },
        {
            "task_id": "T026",
            "category": "resource_conflict",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "Remote_Drive_video_1 和 Telemedicine_video_1 同时抢低时延资源时，先保远程驾驶控制稳定，再保医疗上行，其它视频业务可以让步。",
            "expected_objects": {"flows": ["Remote_Drive_video_1", "Telemedicine_video_1"]},
            "expected_direction": "跨 UE 双关键流分级保护",
            "success_criteria": "Main/IEA/OSA 能保留两个目标和优先级顺序，而不是只优化一个流",
        },
        {
            "task_id": "T027",
            "category": "multi_object_ambiguity",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "AR gaming 那个控制流要更稳，不是视频流；带宽只小幅提高，时延别恶化，mobility 不动。",
            "expected_objects": {
                "app": "AR_Gaming",
                "target_flow": "AR_Gaming_control_1",
                "excluded_flow": "AR_Gaming_video_2",
            },
            "expected_direction": "解析到 AR 控制流并做轻量 QoS 优化",
            "success_criteria": "不把“AR gaming”误绑定到视频流，不改 AM policy",
        },
        {
            "task_id": "T028",
            "category": "multi_object_ambiguity",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "4K 视频里只动主 stream，不动 control；如果为了吞吐要牺牲，先不要碰控制流。",
            "expected_objects": {
                "app": "4K_Video",
                "target_flow": "4K_Video_stream_1",
                "excluded_flow": "4K_Video_control_2",
            },
            "expected_direction": "区分同 app 的 stream 与 control flow",
            "success_criteria": "控制流保持不变，QoS 调整只落在主视频流",
        },
        {
            "task_id": "T029",
            "category": "cross_domain_coordination",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "先把 Remote_Drive_video_1 低时延化；如果 OSA 选择了新切片，就同步 AM allowed/target NSSAI，否则只下发 SM policy，不要凭空改 AM。",
            "expected_objects": {"app": "Remote_Drive", "flow": "Remote_Drive_video_1"},
            "expected_direction": "条件式 QoS-AM 同步",
            "success_criteria": "跨域计划以 optimizer 结果为条件，避免无依据 AM policy 修改",
        },
        {
            "task_id": "T030",
            "category": "cross_domain_coordination",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "Telemedicine_video_1 需要上行保障；若切片变化导致 allowed NSSAI 不包含目标切片，就更新 AM policy，但 Web_Browse 业务不动。",
            "expected_objects": {
                "app": ["Telemedicine", "Web_Browse"],
                "target_flow": "Telemedicine_video_1",
            },
            "expected_direction": "医疗 QoS 优化与必要 AM 同步，排除同 UE 浏览业务",
            "success_criteria": "只在切片选择需要时同步 AM，Web_Browse 不被误调",
        },
        {
            "task_id": "T031",
            "category": "cross_domain_coordination",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "AR_Gaming_control_1 延迟优先；如果需要迁移切片，要保持同 UE 的 AR_Gaming_video_2 业务连续性，不要把视频流一起降配。",
            "expected_objects": {
                "app": "AR_Gaming",
                "target_flow": "AR_Gaming_control_1",
                "excluded_flow": "AR_Gaming_video_2",
            },
            "expected_direction": "控制流低时延优先，同时保护同 UE 视频连续性",
            "success_criteria": "跨流约束被保留，策略不牺牲排除的视频流",
        },
        {
            "task_id": "T032",
            "category": "cross_domain_coordination",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "Cloud_Render_stream_1 要切到更好的资源；如果必须改 UE-AMBR，AMBR 不能低于优化后的 SM 带宽，telemetry 流不动。",
            "expected_objects": {
                "app": "Cloud_Render",
                "target_flow": "Cloud_Render_stream_1",
                "excluded_flow": "Cloud_Render_telemetry_2",
            },
            "expected_direction": "QoS 带宽结果与 AMBR 保持一致",
            "success_criteria": "跨域数值一致性正确，telemetry 流不被纳入策略",
        },
        {
            "task_id": "T033",
            "category": "cross_domain_coordination",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "Factory_Robot_video_1 迁到工业优先资源；如果第一次执行失败，第二轮保持这个 flow 绑定并改成 best-effort QoS，不要转去 IoT_Sensor。",
            "expected_objects": {
                "app": ["Factory_Robot", "IoT_Sensor"],
                "target_flow": "Factory_Robot_video_1",
            },
            "expected_direction": "闭环重试保持目标绑定并降级为 best-effort QoS",
            "success_criteria": "失败反馈进入共享上下文后不发生对象漂移",
        },
        {
            "task_id": "T034",
            "category": "cross_domain_coordination",
            "scenario_ids": ["S2", "S3", "S2P", "S3P"],
            "user_input": "Remote_Drive_video_1 和 Telemedicine_video_1 同时保护；QoS 负责切片和带宽，AM 只在目标切片缺失时同步 allowed NSSAI，不改 RFSP。",
            "expected_objects": {"flows": ["Remote_Drive_video_1", "Telemedicine_video_1"]},
            "expected_direction": "双关键流联合保护，AM 只做最小同步",
            "success_criteria": "跨域职责边界清晰，RFSP 不被误改",
        },
        {
            "task_id": "T035",
            "category": "cross_domain_coordination",
            "scenario_ids": ["S3", "S3P"],
            "user_input": "Drone_Control_video_1 和 Smart_Meter_telemetry_1 拥塞时，保护无人机控制，可降智能电表遥测；若无人机切片变更，再同步 AM policy。",
            "expected_objects": {
                "primary_flow": "Drone_Control_video_1",
                "secondary_flow": "Smart_Meter_telemetry_1",
            },
            "expected_direction": "S3-only 跨 UE 关键控制保护与低优先级遥测降配",
            "success_criteria": "无人机控制优先级高于智能电表遥测，AM 同步有条件触发",
        },
        {
            "task_id": "T036",
            "category": "cross_domain_coordination",
            "scenario_ids": ["S3", "S3P"],
            "user_input": "Web_Browse_session_2 可以牺牲；不能影响 Drone_Control_video_1 和 Factory_Robot_video_1；如果调度失败，重试时保持两个控制流绑定。",
            "expected_objects": {
                "flows": ["Drone_Control_video_1", "Factory_Robot_video_1"],
                "deprioritized_flows": ["Web_Browse_session_2"],
            },
            "expected_direction": "S3-only 多控制流保护与闭环重试绑定保持",
            "success_criteria": "两个关键控制流不丢失，降配对象不会被反向保护",
        },
    ]
    return {
        "meta": {
            "task_count": EXPECTED_TOTAL_COUNT,
            "categories": CATEGORY_ORDER,
        },
        "tasks": tasks,
    }


def _write_task_catalog(path: Path, catalog: Dict[str, Any]) -> None:
    write_json(path, catalog)


def _load_matrix(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_id_list(values: List[str]) -> List[str]:
    return [str(item).strip() for item in values if str(item).strip()]


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _validate_task_inventory(tasks: List[Dict[str, Any]]) -> None:
    if len(tasks) != EXPECTED_TOTAL_COUNT:
        raise ValueError(f"Expected {EXPECTED_TOTAL_COUNT} tasks, got {len(tasks)}")

    category_counts = {category: 0 for category in CATEGORY_ORDER}
    seen_ids: Set[str] = set()
    for task in tasks:
        task_id = str(task.get("task_id") or "").strip()
        category = str(task.get("category") or "").strip()
        if not task_id:
            raise ValueError("Each task must define task_id")
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task_id detected: {task_id}")
        seen_ids.add(task_id)
        if category not in category_counts:
            raise ValueError(f"Task {task_id} uses unsupported category: {category}")
        category_counts[category] += 1

    mismatched = {
        category: category_counts.get(category, 0)
        for category, expected in EXPECTED_CATEGORY_COUNTS.items()
        if category_counts.get(category, 0) != expected
    }
    if mismatched:
        raise ValueError(
            f"Expected category counts {EXPECTED_CATEGORY_COUNTS}, got {mismatched}"
        )


def _build_scenario_inventory(scenario_id: str) -> Dict[str, Any]:
    registry = load_scenario_registry()
    scenario_meta = registry.get(scenario_id)
    if scenario_meta is None:
        raise ValueError(f"Unknown scenario id referenced by task catalog: {scenario_id}")
    source = str(scenario_meta.get("source") or "").strip()
    if not source:
        raise ValueError(f"Scenario {scenario_id} does not define a source file")

    payload = load_yaml_mapping((PROJECT_ROOT / source).resolve())
    apps = payload.get("apps")
    flows = payload.get("flows")
    ues = payload.get("ues")
    if not isinstance(apps, list) or not isinstance(flows, list) or not isinstance(ues, list):
        raise ValueError(f"Scenario {scenario_id} must define apps/flows/ues lists")

    app_names = {str(item.get("name") or "").strip() for item in apps if str(item.get("name") or "").strip()}
    flow_names = {str(item.get("name") or "").strip() for item in flows if str(item.get("name") or "").strip()}
    supis = {str(item.get("supi") or "").strip() for item in ues if str(item.get("supi") or "").strip()}
    flow_supis: Dict[str, str] = {}
    app_supis: Dict[str, Set[str]] = {}
    for item in flows:
        flow_name = str(item.get("name") or "").strip()
        app_name = str(item.get("app_name") or "").strip()
        supi = str(item.get("supi") or "").strip()
        if flow_name and supi:
            flow_supis[flow_name] = supi
        if app_name and supi:
            app_supis.setdefault(app_name, set()).add(supi)

    return {
        "apps": app_names,
        "flows": flow_names,
        "supis": supis,
        "flow_supis": flow_supis,
        "app_supis": app_supis,
    }


def _collect_object_references(expected_objects: Dict[str, Any]) -> Dict[str, Set[str]]:
    refs = {"apps": set(), "flows": set(), "supis": set()}
    if not isinstance(expected_objects, dict):
        raise TypeError("expected_objects must be a mapping")

    for key, value in expected_objects.items():
        normalized_key = str(key).strip()
        values: List[str]
        if isinstance(value, list):
            values = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str):
            values = [value.strip()] if value.strip() else []
        else:
            continue

        if normalized_key == "supi":
            refs["supis"].update(values)
        elif normalized_key == "app":
            refs["apps"].update(values)
        elif normalized_key in {
            "flow",
            "primary_flow",
            "secondary_flow",
            "target_flow",
            "excluded_flow",
            "protected_flow",
        }:
            refs["flows"].update(values)
        elif normalized_key in {"flows", "priority_flows", "deprioritized_flows"}:
            refs["flows"].update(values)
        elif normalized_key == "deprioritized_app":
            refs["apps"].update(values)
    return refs


def _validate_tasks_against_scenarios(tasks: List[Dict[str, Any]]) -> None:
    inventory_cache: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        task_id = str(task.get("task_id") or "").strip()
        scenario_ids = _normalize_id_list(list(task.get("scenario_ids") or []))
        if not scenario_ids:
            raise ValueError(f"Task {task_id} must declare at least one scenario_id")
        expected_objects = task.get("expected_objects", {})
        refs = _collect_object_references(expected_objects)
        for scenario_id in scenario_ids:
            if scenario_id not in inventory_cache:
                inventory_cache[scenario_id] = _build_scenario_inventory(scenario_id)
            inventory = inventory_cache[scenario_id]
            for obj_type in ("apps", "flows", "supis"):
                missing = sorted(refs[obj_type] - inventory[obj_type])
                if missing:
                    raise ValueError(
                        f"Task {task_id} references missing {obj_type[:-1]} values in {scenario_id}: {missing}"
                    )


def _resolve_allowed_scenarios(experiment_id: str) -> List[str]:
    if not experiment_id:
        return []
    payload = _load_matrix(MATRIX_PATH)
    for item in payload.get("experiments", []):
        if str(item.get("id") or "").strip() == experiment_id:
            return _normalize_id_list(list(item.get("scenarios") or []))
    raise ValueError(f"Unknown experiment id: {experiment_id}")


def _iter_expected_values(expected_objects: Dict[str, Any], keys: Iterable[str]) -> List[str]:
    target_keys = set(keys)
    values: List[str] = []
    for key, value in expected_objects.items():
        if str(key).strip() not in target_keys:
            continue
        if isinstance(value, list):
            values.extend(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def _resolve_task_supis(task: Dict[str, Any], *, resolved_scenario_id: str) -> List[str]:
    expected_objects = task.get("expected_objects", {})
    if not isinstance(expected_objects, dict):
        return []

    inventory = _build_scenario_inventory(resolved_scenario_id)
    flow_supis: Dict[str, str] = inventory["flow_supis"]
    app_supis: Dict[str, Set[str]] = inventory["app_supis"]
    supis: List[str] = _iter_expected_values(expected_objects, {"supi"})

    flow_keys = {
        "flow",
        "primary_flow",
        "secondary_flow",
        "target_flow",
        "excluded_flow",
        "protected_flow",
        "flows",
        "priority_flows",
        "deprioritized_flows",
    }
    for flow_name in _iter_expected_values(expected_objects, flow_keys):
        supi = flow_supis.get(flow_name)
        if supi:
            supis.append(supi)

    for app_name in _iter_expected_values(expected_objects, {"app", "deprioritized_app"}):
        supis.extend(sorted(app_supis.get(app_name, set())))

    return _dedupe_preserve_order(supis)


def _with_supi_hint(user_input: str, supis: List[str]) -> str:
    missing_supis = [supi for supi in supis if supi not in user_input]
    if not missing_supis:
        return user_input
    return f"{user_input}（supi: {', '.join(missing_supis)}）"


def _build_record(index: int, task: Dict[str, Any], *, resolved_scenario_id: str) -> Dict[str, Any]:
    user_input = str(task["user_input"]).strip()
    scenario_ids = [str(item).strip() for item in task.get("scenario_ids", []) if str(item).strip()]
    category = str(task.get("category") or "").strip()
    task_id = str(task.get("task_id") or f"T{index:03d}").strip()
    if not resolved_scenario_id:
        raise ValueError(f"Task {task_id} is missing a resolved scenario_id")
    user_input = _with_supi_hint(user_input, _resolve_task_supis(task, resolved_scenario_id=resolved_scenario_id))
    return {
        "record_index": index,
        "user_input": user_input,
        "messages": [{"role": "user", "content": user_input}],
        "context": "",
        "scenario_id": resolved_scenario_id,
        "scenario_tags": _dedupe_preserve_order(["experiment", category, resolved_scenario_id, *scenario_ids]),
        "task_metadata": {
            "task_id": task_id,
            "category": category,
            "expected_objects": task.get("expected_objects", {}),
            "expected_direction": task.get("expected_direction", ""),
            "success_criteria": task.get("success_criteria", ""),
            "scenario_ids": scenario_ids,
        },
    }


def _filter_tasks(
    tasks: List[Dict[str, Any]],
    *,
    scenario_id: str,
    experiment_id: str,
) -> List[Dict[str, Any]]:
    allowed_scenarios = set(_resolve_allowed_scenarios(experiment_id)) if experiment_id else set()
    explicit_scenario = str(scenario_id or "").strip()
    filtered: List[Dict[str, Any]] = []
    for task in tasks:
        task_scenarios = {
            str(item).strip()
            for item in (task.get("scenario_ids") or [])
            if str(item).strip()
        }
        if explicit_scenario and explicit_scenario not in task_scenarios:
            continue
        if allowed_scenarios and not (task_scenarios & allowed_scenarios):
            continue
        filtered.append(task)
    return filtered


def _resolve_record_scenario_id(task: Dict[str, Any], *, explicit_scenario_id: str) -> str:
    task_id = str(task.get("task_id") or "").strip()
    scenario_ids = _normalize_id_list(list(task.get("scenario_ids") or []))
    if explicit_scenario_id:
        if explicit_scenario_id not in scenario_ids:
            raise ValueError(f"Task {task_id} does not belong to scenario {explicit_scenario_id}")
        return explicit_scenario_id
    if len(scenario_ids) == 1:
        return scenario_ids[0]
    raise ValueError(
        f"Task {task_id} maps to multiple scenarios {scenario_ids}. "
        "Run build_user_inputs.py with --scenario to generate unambiguous records."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build experiment user inputs from the canonical 36-task catalog.")
    parser.add_argument("--scenario", default="", help="Filter tasks by scenario id, e.g. S2")
    parser.add_argument("--experiment", default="", help="Filter tasks by experiment id, e.g. E1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = _build_task_catalog()
    tasks: List[Dict[str, Any]] = list(catalog.get("tasks", []))
    _validate_task_inventory(tasks)
    _validate_tasks_against_scenarios(tasks)
    _write_task_catalog(TASK_CATALOG_PATH, catalog)

    filtered_tasks = _filter_tasks(
        tasks,
        scenario_id=str(args.scenario or "").strip(),
        experiment_id=str(args.experiment or "").strip(),
    )
    if not filtered_tasks:
        raise RuntimeError("No tasks matched the requested experiment/scenario filters.")

    explicit_scenario_id = str(args.scenario or "").strip()
    records = [
        _build_record(
            index,
            task,
            resolved_scenario_id=_resolve_record_scenario_id(task, explicit_scenario_id=explicit_scenario_id),
        )
        for index, task in enumerate(filtered_tasks, start=1)
    ]
    payload = {
        "meta": {
            "count": len(records),
            "catalog_count": len(tasks),
            "source": str(TASK_CATALOG_PATH),
            "experiment_id": str(args.experiment or "").strip(),
            "scenario_id": str(args.scenario or "").strip(),
            "categories": CATEGORY_ORDER,
        },
        "records": records,
    }
    output_path = scoped_catalog_input_path(
        experiment_id=str(args.experiment or "").strip(),
        scenario_id=str(args.scenario or "").strip(),
    )
    write_json(output_path, payload)

    default_output = default_catalog_input_path()
    write_json(default_output, payload)
    print(f"Wrote {len(records)} experiment records -> {output_path}")
    print(f"Canonical catalog size: {len(tasks)} tasks across {len(CATEGORY_ORDER)} categories")
    print(f"Updated default input file -> {default_output}")


if __name__ == "__main__":
    main()
