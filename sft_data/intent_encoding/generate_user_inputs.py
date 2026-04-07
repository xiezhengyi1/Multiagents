from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.tools.db_tool import get_latest_snapshot_data


DEFAULT_TARGET_SUCCESS_RATE = 0.8
DEFAULT_SAMPLE_COUNT = 30
DEFAULT_SEED = 42

FLOW_KIND_TO_CN = {
    "video": "视频流",
    "control": "控制流",
    "telemetry": "遥测流",
    "other": "业务流",
}

SERVICE_TYPE_TO_CN = {
    "eMBB": "大带宽体验",
    "URLLC": "低时延稳定性",
    "mMTC": "海量连接稳定性",
}


@dataclass(frozen=True)
class UserInputCandidate:
    user_input: str
    messages: tuple[Dict[str, str], ...]
    context: str
    supi: str
    app_name: str
    flow_name: str
    flow_id: str
    slice_snssai: str
    slice_name: str
    service_type: str
    template_family: str
    difficulty: str
    estimated_success_probability: float
    rationale: str
    scenario_tags: tuple[str, ...]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_network_state(input_path: Optional[Path]) -> Dict[str, Any]:
    if input_path is not None:
        payload = _load_json(input_path)
        if "scenario" in payload and isinstance(payload["scenario"], dict):
            return payload
        if all(key in payload for key in ("apps", "slices", "nodes")):
            return {"scenario": payload}
        raise ValueError("Input JSON must contain either top-level 'scenario' or direct 'apps/slices/nodes'.")

    snapshot = get_latest_snapshot_data()
    if not snapshot:
        raise RuntimeError("No latest network snapshot is available. Provide --input explicitly.")
    return {"scenario": snapshot}


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _looks_mojibake(text: str) -> bool:
    return any(marker in text for marker in ("锟", "�", "鏈", "闄", "绛栫", "瑙"))


def infer_flow_kind(flow_name: str) -> str:
    lowered = _normalized_lower(flow_name)
    if "video" in lowered:
        return "video"
    if "control" in lowered:
        return "control"
    if "telemetry" in lowered:
        return "telemetry"
    return "other"


def _slice_display_name(slice_name: str, slice_snssai: str) -> str:
    return slice_name or slice_snssai or "当前切片"


def _single_turn_messages(user_input: str) -> tuple[Dict[str, str], ...]:
    return ({"role": "user", "content": user_input},)


def _follow_up_messages(
    *,
    lead_user: str,
    assistant_reply: str,
    final_user: str,
) -> tuple[Dict[str, str], ...]:
    # 中文标注：构造多轮上下文，增强训练样本里的对话复杂度。
    return (
        {"role": "user", "content": lead_user},
        {"role": "assistant", "content": assistant_reply},
        {"role": "user", "content": final_user},
    )


def _merge_context_blocks(*blocks: str) -> str:
    return "\n\n".join(str(block).strip() for block in blocks if str(block or "").strip())


def _build_memory_context(*entries: tuple[str, str]) -> str:
    lines = [f"{role}: {content}" for role, content in entries if str(role).strip() and str(content).strip()]
    if not lines:
        return ""
    return "[Memory][Short-Term]\n" + "\n".join(lines)


def _build_feedback_context(
    *,
    round_index: int,
    performance_metrics: Dict[str, Any],
    violation_details: str,
    correction_suggestion: str,
) -> str:
    return (
        f"[PDA Feedback][Round {round_index}]\n"
        "execution_status: Failed\n"
        f"performance_metrics: {json.dumps(performance_metrics, ensure_ascii=False)}\n"
        f"violation_details: {violation_details}\n"
        f"correction_suggestion: {correction_suggestion}\n"
        "Use this feedback to refine the next round of intent understanding."
    )


def build_slice_index(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    scenario = payload.get("scenario", {})
    slices = scenario.get("slices", []) if isinstance(scenario, dict) else []
    stats = payload.get("slice_stats", []) if isinstance(payload.get("slice_stats"), list) else []

    by_snssai: Dict[str, Dict[str, Any]] = {}
    for item in slices:
        if not isinstance(item, dict):
            continue
        snssai = str(item.get("snssai") or "").strip()
        if not snssai:
            continue
        by_snssai[snssai] = dict(item)

    for item in stats:
        if not isinstance(item, dict):
            continue
        snssai = str(item.get("SNSSAI") or "").strip()
        if not snssai:
            continue
        target = by_snssai.setdefault(snssai, {})
        target.update(
            {
                "slice_stats_name": item.get("Slice"),
                "load_ul_pct": _safe_float(item.get("Load UL (%)")),
                "load_dl_pct": _safe_float(item.get("Load DL (%)")),
                "rem_ul": _safe_float(item.get("Rem UL (M)")),
                "rem_dl": _safe_float(item.get("Rem DL (M)")),
                "cap": item.get("Cap UL/DL (M)"),
                "alloc_ul": _safe_float(item.get("Alloc UL (M)")),
                "alloc_dl": _safe_float(item.get("Alloc DL (M)")),
            }
        )

    return by_snssai


def iter_flow_contexts(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    scenario = payload.get("scenario", {})
    apps = scenario.get("apps", []) if isinstance(scenario, dict) else []
    slice_index = build_slice_index(payload)

    for app in apps:
        if not isinstance(app, dict):
            continue
        supi = str(app.get("supi") or "").strip()
        app_name = str(app.get("name") or "").strip()
        app_id = str(app.get("app_id") or "").strip()
        flows = app.get("flows") or []
        flow_names = [str(item.get("name") or "").strip() for item in flows if isinstance(item, dict)]
        for flow in flows:
            if not isinstance(flow, dict):
                continue
            flow_name = str(flow.get("name") or "").strip()
            slice_snssai = str(flow.get("old_slice") or "").strip()
            yield {
                "supi": supi,
                "app_name": app_name,
                "app_id": app_id,
                "flow": flow,
                "flow_count": len(flows),
                "sibling_flow_names": [item for item in flow_names if item and item != flow_name],
                "slice": slice_index.get(slice_snssai, {}),
                "slice_snssai": slice_snssai,
            }


def analyze_issue(context: Dict[str, Any]) -> Dict[str, str]:
    flow = context["flow"]
    slice_info = context["slice"]
    service_type = str(flow.get("service_type") or "").strip()
    req_ul = _safe_float(flow.get("bw_ul")) or 0.0
    req_dl = _safe_float(flow.get("bw_dl")) or 0.0
    current_ul = _safe_float(flow.get("old_allocated_bw_ul"))
    current_dl = _safe_float(flow.get("old_allocated_bw_dl"))
    sim_loss = _safe_float(flow.get("sim_loss_rate")) or 0.0
    sim_jitter = _safe_float(flow.get("sim_jitter")) or 0.0
    lat = _safe_float(flow.get("lat")) or 0.0
    load_ul = _safe_float(slice_info.get("load_ul_pct")) or 0.0
    load_dl = _safe_float(slice_info.get("load_dl_pct")) or 0.0
    rem_ul = _safe_float(slice_info.get("rem_ul"))
    rem_dl = _safe_float(slice_info.get("rem_dl"))

    downlink_gap = current_dl is not None and req_dl > 0 and current_dl < req_dl
    uplink_gap = current_ul is not None and req_ul > 0 and current_ul < req_ul
    dl_congested = load_dl >= 85 or (rem_dl is not None and rem_dl <= 0)
    ul_congested = load_ul >= 85 or (rem_ul is not None and rem_ul <= 0)
    high_loss = sim_loss >= 5
    high_jitter = sim_jitter >= 5

    if downlink_gap or dl_congested:
        return {
            "direction": "downlink",
            "action_cn": "提升下行体验",
            "pain_cn": "下行带宽明显不足",
            "rationale": "slice 下行负载偏高或当前下行分配低于请求值",
        }
    if uplink_gap or ul_congested:
        return {
            "direction": "uplink",
            "action_cn": "增强上行稳定性",
            "pain_cn": "上行资源偏紧",
            "rationale": "slice 上行负载偏高或当前上行分配低于请求值",
        }
    if service_type == "URLLC" or lat <= 20:
        return {
            "direction": "latency",
            "action_cn": "降低时延并保证稳定性",
            "pain_cn": "时延敏感业务需要更稳",
            "rationale": "业务属于低时延或超低时延类型",
        }
    if high_loss or high_jitter:
        return {
            "direction": "stability",
            "action_cn": "提升稳定性",
            "pain_cn": "丢包或抖动偏高",
            "rationale": "仿真指标显示丢包率或抖动偏高",
        }
    return {
        "direction": "experience",
        "action_cn": "优化整体体验",
        "pain_cn": "希望业务体验更平滑",
        "rationale": "当前资源并非极端紧张，适合做体验型诉求",
    }


def analyze_topology_signals(context: Dict[str, Any]) -> Dict[str, str | bool]:
    slice_info = context["slice"]
    slice_snssai = context["slice_snssai"]
    slice_name = str(slice_info.get("name") or slice_info.get("slice_stats_name") or "").strip()
    load_ul = _safe_float(slice_info.get("load_ul_pct")) or 0.0
    load_dl = _safe_float(slice_info.get("load_dl_pct")) or 0.0
    latency_sla = _safe_float(slice_info.get("latency"))

    overloaded = load_ul >= 85 or load_dl >= 85
    if load_dl >= 90:
        pressure_cn = "下行已经接近打满"
    elif load_ul >= 90:
        pressure_cn = "上行已经接近打满"
    elif overloaded:
        pressure_cn = "资源已经比较紧张"
    elif load_dl >= 70 or load_ul >= 70:
        pressure_cn = "负载正在走高"
    else:
        pressure_cn = "负载还算平稳"

    if latency_sla is None:
        latency_cn = "当前 SLA 信息未显式标注"
    elif latency_sla <= 5:
        latency_cn = "SLA 对时延极其敏感"
    elif latency_sla <= 20:
        latency_cn = "SLA 偏向低时延"
    else:
        latency_cn = "SLA 更偏向吞吐与稳定性平衡"

    return {
        "slice_display": _slice_display_name(slice_name, slice_snssai),
        "pressure_cn": pressure_cn,
        "latency_cn": latency_cn,
        "overloaded": overloaded,
        "reroute_cn": "必要时考虑换到更合适的切片" if overloaded else "优先在当前切片内优化",
    }


def estimate_success_probability(
    *,
    include_exact_app: bool,
    include_exact_flow: bool,
    include_direction_keyword: bool,
    ambiguous_flow_name: bool,
    mojibake_anchor: bool,
    flow_count: int,
    difficulty: str,
) -> float:
    score = 0.15
    score += 0.45  # explicit supi is always included
    if include_exact_app:
        score += 0.15
    if include_exact_flow:
        score += 0.17
    if include_direction_keyword:
        score += 0.05
    if ambiguous_flow_name and not include_exact_flow:
        score -= 0.18
    if flow_count > 1 and not include_exact_flow:
        score -= 0.08
    if mojibake_anchor:
        score -= 0.12
    if difficulty == "medium":
        score -= 0.10
    elif difficulty == "low":
        score -= 0.28
    return max(0.08, min(0.97, round(score, 3)))


def build_templates(context: Dict[str, Any]) -> List[UserInputCandidate]:
    supi = context["supi"]
    app_name = context["app_name"]
    flow = context["flow"]
    slice_info = context["slice"]
    flow_name = str(flow.get("name") or "").strip()
    flow_id = str(flow.get("flow_id") or "").strip()
    slice_snssai = context["slice_snssai"]
    slice_name = str(slice_info.get("name") or slice_info.get("slice_stats_name") or "").strip()
    service_type = str(flow.get("service_type") or "").strip()
    flow_kind = infer_flow_kind(flow_name)
    flow_kind_cn = FLOW_KIND_TO_CN[flow_kind]
    issue = analyze_issue(context)
    mojibake_anchor = _looks_mojibake(flow_name) or _looks_mojibake(app_name)
    ambiguous_flow_name = context["flow_count"] > 1 and flow_kind in {"video", "control", "telemetry"}
    service_cn = SERVICE_TYPE_TO_CN.get(service_type, "业务体验")
    topology = analyze_topology_signals(context)
    sibling_flow_names = [item for item in context.get("sibling_flow_names", []) if item]
    sibling_flow_name = sibling_flow_names[0] if sibling_flow_names else ""

    if not supi or not app_name or not flow_name:
        return []

    candidates: List[UserInputCandidate] = []

    memory_context = _build_memory_context(
        ("user", f"优先保障{supi}的{app_name}/{flow_name}，同时别误伤其他业务。"),
        ("IEA", f"resolved flow={flow_name}, flow_id={flow_id}, service_type={service_type or 'unknown'}"),
    )
    retry_context = _merge_context_blocks(
        memory_context,
        _build_feedback_context(
            round_index=1,
            performance_metrics={"dispatch_results": [], "assurance_results": []},
            violation_details=(
                f"{topology['slice_display']}{topology['pressure_cn']}，原执行方案可能影响"
                f"{sibling_flow_name or '同应用其他流'}。"
            ),
            correction_suggestion=(
                f"Refine the intent so it protects only {app_name}/{flow_name}, avoids slice expansion, "
                f"and respects {topology['latency_cn']}."
            ),
        ),
    )

    def add_candidate(
        *,
        user_input: str,
        family: str,
        exact_app: bool,
        exact_flow: bool,
        direction_kw: bool,
        difficulty: str,
        messages: tuple[Dict[str, str], ...] | None = None,
        request_context: str = "",
        probability_bias: float = 0.0,
        scenario_tags: tuple[str, ...] = (),
    ) -> None:
        probability = estimate_success_probability(
            include_exact_app=exact_app,
            include_exact_flow=exact_flow,
            include_direction_keyword=direction_kw,
            ambiguous_flow_name=ambiguous_flow_name,
            mojibake_anchor=mojibake_anchor,
            flow_count=context["flow_count"],
            difficulty=difficulty,
        )
        probability = max(0.08, min(0.97, round(probability + probability_bias, 3)))
        candidates.append(
            UserInputCandidate(
                user_input=user_input,
                messages=messages or _single_turn_messages(user_input),
                context=request_context,
                supi=supi,
                app_name=app_name,
                flow_name=flow_name,
                flow_id=flow_id,
                slice_snssai=slice_snssai,
                slice_name=slice_name,
                service_type=service_type,
                template_family=family,
                difficulty=difficulty,
                estimated_success_probability=probability,
                rationale=f"{issue['rationale']}；{topology['pressure_cn']}；模板难度={difficulty}",
                scenario_tags=tuple(
                    sorted(
                        {
                            *scenario_tags,
                            difficulty,
                            issue['direction'],
                            flow_kind,
                            *(('coordinator_context',) if request_context else ()),
                        }
                    )
                ),
            )
        )

    # 中文标注：高置信度样本保留精确锚点，同时加入拓扑约束与多流约束。
    add_candidate(
        user_input=f"我想优化{app_name}里的{flow_name}，重点是{issue['action_cn']}，supi:{supi}",
        family="anchor_exact",
        exact_app=True,
        exact_flow=True,
        direction_kw=issue["direction"] in {"uplink", "downlink"},
        difficulty="high",
        scenario_tags=("exact_anchor", "single_turn"),
    )
    add_candidate(
        user_input=f"{app_name}这个应用里，{flow_name}这条流最近{issue['pain_cn']}，请优先处理，supi:{supi}",
        family="pain_statement",
        exact_app=True,
        exact_flow=True,
        direction_kw=False,
        difficulty="high",
        scenario_tags=("pain_statement", "single_turn"),
    )
    add_candidate(
        user_input=f"请针对{supi}的{app_name}/{flow_name}做优化，我更关注{service_cn}，supi:{supi}",
        family="path_anchor",
        exact_app=True,
        exact_flow=True,
        direction_kw=False,
        difficulty="high",
        scenario_tags=("service_expectation", "single_turn"),
    )
    topology_guardrail_input = (
        f"当前{topology['slice_display']}{topology['pressure_cn']}，但{supi}的{app_name}/{flow_name}不能继续受影响。"
        f"请优先保障这条流，{topology['reroute_cn']}，并且不要误伤同应用其他流。"
    )
    add_candidate(
        user_input=topology_guardrail_input,
        family="topology_guardrail",
        exact_app=True,
        exact_flow=True,
        direction_kw=True,
        difficulty="high",
        messages=_follow_up_messages(
            lead_user=f"我们在排查 {topology['slice_display']} 的资源波动。",
            assistant_reply="可以，告诉我需要优先保障的 UE 或具体业务流。",
            final_user=topology_guardrail_input,
        ),
        request_context=retry_context,
        probability_bias=-0.03,
        scenario_tags=("topology", "multi_turn", "guardrail", "feedback_retry"),
    )
    flow_id_input = (
        f"请按 flow_id={flow_id} 定位 {flow_name}，只优化这一条，目标是{issue['action_cn']}；"
        f"如果{topology['slice_display']}资源吃紧，先给出不扩容的优化意图。supi:{supi}"
    )
    add_candidate(
        user_input=flow_id_input,
        family="flow_id_guardrail",
        exact_app=True,
        exact_flow=True,
        direction_kw=True,
        difficulty="high",
        request_context=memory_context,
        probability_bias=-0.02,
        scenario_tags=("flow_id", "guardrail", "single_turn", "memory_replay"),
    )

    if sibling_flow_name:
        sibling_input = (
            f"{app_name}里先处理{flow_name}，{sibling_flow_name}先不要动。{issue['pain_cn']}，"
            f"如果当前切片资源不够，请优先保这条流，supi:{supi}"
        )
        add_candidate(
            user_input=sibling_input,
            family="sibling_disambiguation",
            exact_app=True,
            exact_flow=True,
            direction_kw=False,
            difficulty="high",
            probability_bias=-0.03,
            scenario_tags=("multi_flow", "guardrail", "single_turn"),
        )
        comparative_input = (
            f"{app_name}里{flow_name}比{sibling_flow_name}更关键，先保证前者的{service_cn}，"
            f"再考虑其他流，supi:{supi}"
        )
        add_candidate(
            user_input=comparative_input,
            family="comparative_priority",
            exact_app=True,
            exact_flow=True,
            direction_kw=False,
            difficulty="medium",
            probability_bias=-0.02,
            scenario_tags=("multi_flow", "relative_priority", "single_turn"),
        )

    add_candidate(
        user_input=f"我想让{app_name}的{flow_kind_cn}更稳定，supi:{supi}",
        family="kind_anchor",
        exact_app=True,
        exact_flow=False,
        direction_kw=False,
        difficulty="medium",
        scenario_tags=("kind_anchor", "single_turn"),
    )
    add_candidate(
        user_input=f"{app_name}体验不理想，尤其是{flow_kind_cn}部分，麻烦优化一下，supi:{supi}",
        family="experience_request",
        exact_app=True,
        exact_flow=False,
        direction_kw=False,
        difficulty="medium",
        scenario_tags=("experience", "single_turn"),
    )
    slice_aware_input = (
        f"{app_name}现在跑在{topology['slice_display']}，这边{topology['pressure_cn']}。"
        f"先围绕{flow_kind_cn}给个优化意图，supi:{supi}"
    )
    add_candidate(
        user_input=slice_aware_input,
        family="slice_aware",
        exact_app=True,
        exact_flow=False,
        direction_kw=True,
        difficulty="medium",
        request_context=memory_context,
        probability_bias=-0.03,
        scenario_tags=("topology", "slice_pressure", "single_turn", "memory_replay"),
    )
    policy_boundary_input = (
        f"先别直接扩容，看看能不能只通过 QoS 或优先级调整让{app_name}的{flow_kind_cn}更稳，"
        f"{topology['latency_cn']}，supi:{supi}"
    )
    add_candidate(
        user_input=policy_boundary_input,
        family="policy_boundary",
        exact_app=True,
        exact_flow=False,
        direction_kw=False,
        difficulty="medium",
        probability_bias=-0.04,
        scenario_tags=("qos_boundary", "single_turn"),
    )
    followup_input = f"就是{supi}这条 UE 的{app_name}/{flow_name}，先{issue['action_cn']}，不要影响其他应用。"
    add_candidate(
        user_input=followup_input,
        family="multi_turn_followup",
        exact_app=True,
        exact_flow=True,
        direction_kw=True,
        difficulty="medium",
        messages=_follow_up_messages(
            lead_user=f"我们刚看过{topology['slice_display']}的拓扑，想先稳住一条关键业务。",
            assistant_reply="明白，请告诉我具体 UE、应用或业务流。",
            final_user=followup_input,
        ),
        request_context=retry_context,
        probability_bias=-0.06,
        scenario_tags=("multi_turn", "topology", "guardrail", "feedback_retry"),
    )

    add_candidate(
        user_input=f"这个用户的业务体验太差了，想提一下{flow_kind_cn}质量，supi:{supi}",
        family="weak_anchor",
        exact_app=False,
        exact_flow=False,
        direction_kw=False,
        difficulty="low",
        scenario_tags=("weak_anchor", "single_turn"),
    )
    add_candidate(
        user_input=f"帮我把网络体验调好一点，最好照顾下{flow_kind_cn}，supi:{supi}",
        family="generic_request",
        exact_app=False,
        exact_flow=False,
        direction_kw=False,
        difficulty="low",
        scenario_tags=("generic", "single_turn"),
    )
    add_candidate(
        user_input=f"{topology['slice_display']}这边感觉有点堵，麻烦帮我照顾一下{supi}那个{flow_kind_cn}业务。",
        family="slice_vague",
        exact_app=False,
        exact_flow=False,
        direction_kw=True,
        difficulty="low",
        request_context=retry_context,
        probability_bias=-0.03,
        scenario_tags=("topology", "vague", "single_turn", "feedback_retry"),
    )
    add_candidate(
        user_input=f"{service_cn}这类业务最近不稳，优先看一下这个 UE 的{flow_kind_cn}，先别大动别的业务，supi:{supi}",
        family="service_vague",
        exact_app=False,
        exact_flow=False,
        direction_kw=False,
        difficulty="low",
        probability_bias=-0.04,
        scenario_tags=("service_only", "vague", "single_turn"),
    )
    return candidates


def generate_candidate_pool(payload: Dict[str, Any]) -> List[UserInputCandidate]:
    pool: List[UserInputCandidate] = []
    for context in iter_flow_contexts(payload):
        pool.extend(build_templates(context))
    if not pool:
        raise RuntimeError("No valid app/flow/supi triples were found in the network state.")
    return pool


def _select_candidates_by_difficulty(
    pool: Sequence[UserInputCandidate],
    *,
    total_count: int,
    target_success_rate: float,
    seed: int,
) -> List[UserInputCandidate]:
    rng = random.Random(seed)
    buckets = {
        "high": [item for item in pool if item.difficulty == "high"],
        "medium": [item for item in pool if item.difficulty == "medium"],
        "low": [item for item in pool if item.difficulty == "low"],
    }
    for items in buckets.values():
        rng.shuffle(items)

    assumed_high = 0.90
    assumed_medium = 0.68
    assumed_low = 0.38
    low_share = 0.10
    high_share = (target_success_rate - assumed_medium * (1 - low_share) - assumed_low * low_share) / (
        assumed_high - assumed_medium
    )
    high_share = max(0.35, min(0.80, high_share))
    medium_share = max(0.10, 1.0 - high_share - low_share)

    raw_desired = {
        "high": total_count * high_share,
        "medium": total_count * medium_share,
        "low": total_count * low_share,
    }
    desired = {key: int(math.floor(value)) for key, value in raw_desired.items()}
    for key in ("high", "medium", "low"):
        if desired[key] == 0 and buckets[key]:
            desired[key] = 1

    current_total = sum(desired.values())
    remainders = sorted(
        ((raw_desired[key] - desired[key], key) for key in ("high", "medium", "low")),
        reverse=True,
    )
    while current_total < total_count:
        for _, key in remainders:
            desired[key] += 1
            current_total += 1
            if current_total >= total_count:
                break
    while current_total > total_count:
        for _, key in reversed(remainders):
            if desired[key] > 1:
                desired[key] -= 1
                current_total -= 1
                if current_total <= total_count:
                    break

    selected: List[UserInputCandidate] = []
    used_inputs: set[str] = set()

    def drain(items: List[UserInputCandidate], limit: int) -> int:
        added = 0
        for item in items:
            if added >= limit:
                break
            if item.user_input in used_inputs:
                continue
            selected.append(item)
            used_inputs.add(item.user_input)
            added += 1
        return added

    counts = {key: drain(buckets[key], desired[key]) for key in ("high", "medium", "low")}

    remaining = total_count - len(selected)
    if remaining > 0:
        combined = list(pool)
        rng.shuffle(combined)
        drain(combined, remaining)

    if len(selected) > total_count:
        selected = selected[:total_count]

    return selected


def build_output(
    payload: Dict[str, Any],
    *,
    count: int,
    target_success_rate: float,
    seed: int,
) -> Dict[str, Any]:
    pool = generate_candidate_pool(payload)
    selected = _select_candidates_by_difficulty(
        pool,
        total_count=count,
        target_success_rate=target_success_rate,
        seed=seed,
    )
    estimated_success_rate = round(
        sum(item.estimated_success_probability for item in selected) / max(1, len(selected)),
        4,
    )

    return {
        "meta": {
            "count": len(selected),
            "target_success_rate": target_success_rate,
            "estimated_success_rate": estimated_success_rate,
            "seed": seed,
            "source_pool_size": len(pool),
        },
        "records": [
            {
                "user_input": item.user_input,
                "messages": list(item.messages),
                "context": item.context,
                "supi": item.supi,
                "app_name": item.app_name,
                "flow_name": item.flow_name,
                "flow_id": item.flow_id,
                "slice_snssai": item.slice_snssai,
                "slice_name": item.slice_name,
                "service_type": item.service_type,
                "template_family": item.template_family,
                "difficulty": item.difficulty,
                "estimated_success_probability": item.estimated_success_probability,
                "rationale": item.rationale,
                "scenario_tags": list(item.scenario_tags),
            }
            for item in selected
        ],
    }


def _write_output(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for row in payload["records"]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diversified Chinese user inputs from current network state.")
    parser.add_argument("--input", type=Path, default=None, help="Path to a network-state JSON file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "generated_user_inputs.json",
        help="Output path. Supports .json and .jsonl.",
    )
    parser.add_argument("--count", type=int, default=DEFAULT_SAMPLE_COUNT, help="Number of user inputs to generate.")
    parser.add_argument(
        "--target-success-rate",
        type=float,
        default=DEFAULT_TARGET_SUCCESS_RATE,
        help="Target average estimated execution success rate.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive.")
    if not 0.0 < args.target_success_rate < 1.0:
        raise ValueError("--target-success-rate must be between 0 and 1.")

    network_state = load_network_state(args.input)
    payload = build_output(
        network_state,
        count=args.count,
        target_success_rate=args.target_success_rate,
        seed=args.seed,
    )
    _write_output(args.output, payload)

    meta = payload["meta"]
    print(f"Generated {meta['count']} user inputs -> {args.output}")
    print(f"Estimated success rate: {meta['estimated_success_rate']:.4f} (target={meta['target_success_rate']:.4f})")


if __name__ == "__main__":
    main()
