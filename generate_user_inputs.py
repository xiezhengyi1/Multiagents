from __future__ import annotations

import argparse
import json
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
DEFAULT_SAMPLE_COUNT = 50
DEFAULT_SEED = 42


@dataclass(frozen=True)
class UserInputCandidate:
    user_input: str
    messages: tuple[Dict[str, str], ...]


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


def _single_turn_messages(user_input: str) -> tuple[Dict[str, str], ...]:
    return ({"role": "user", "content": user_input},)


def _infer_flow_hint(flow_name: str) -> str:
    lowered = flow_name.lower()
    if "video" in lowered or "视频" in flow_name:
        return "视频流"
    if "control" in lowered or "控制" in flow_name:
        return "控制流"
    if "iot" in lowered or "telemetry" in lowered or "传感" in flow_name:
        return "物联网流"
    return "业务流"


def iter_flow_anchors(payload: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    scenario = payload.get("scenario", {})
    apps = scenario.get("apps", []) if isinstance(scenario, dict) else []

    for app in apps:
        if not isinstance(app, dict):
            continue
        supi = str(app.get("supi") or "").strip()
        app_name = str(app.get("name") or app.get("app_id") or "").strip()
        if not supi or not app_name:
            continue

        for flow in app.get("flows") or []:
            if not isinstance(flow, dict):
                continue
            flow_name = str(flow.get("name") or flow.get("flow_id") or "").strip()
            if not flow_name:
                continue
            yield {
                "supi": supi,
                "app_name": app_name,
                "flow_name": flow_name,
            }


def build_templates(anchor: Dict[str, str]) -> List[UserInputCandidate]:
    supi = anchor["supi"]
    app_name = anchor["app_name"]
    flow_name = anchor["flow_name"]
    flow_hint = _infer_flow_hint(flow_name)

    # 中文标注：混合精确信息、缺失字段、模糊语义和歧义表达，模拟更真实的用户输入。
    user_inputs = [
        f"请优化{supi}的{app_name}/{flow_name}",
        f"请保障{supi}上{app_name}里的{flow_name}体验",
        f"把{supi}这个 UE 的{app_name}中{flow_name}调优一下",
        f"{supi}这个用户的{app_name}体验不太好，帮我看下",
        f"帮我看看{app_name}这个业务，最近感觉不太稳定",
        f"这个{flow_hint}是不是有点问题，帮我优化一下",
        f"刚才那个{app_name}里的{flow_name}再调一下",
        f"把{flow_name}处理一下，别影响其他业务",
        f"我想优先保住{app_name}，具体哪条流你先判断一下",
        f"这个 UE 上有个业务不顺，像是{app_name}那边的问题",
        f"先别大改，看看{supi}这边那个{flow_hint}能不能稳一点",
        f"{app_name}里可能不是所有流都有问题，你先判断是不是{flow_name}",
    ]

    return [
        UserInputCandidate(
            user_input=user_input,
            messages=_single_turn_messages(user_input),
        )
        for user_input in user_inputs
    ]


def generate_candidate_pool(payload: Dict[str, Any]) -> List[UserInputCandidate]:
    pool: List[UserInputCandidate] = []
    seen_inputs: set[str] = set()

    for anchor in iter_flow_anchors(payload):
        for candidate in build_templates(anchor):
            if candidate.user_input in seen_inputs:
                continue
            seen_inputs.add(candidate.user_input)
            pool.append(candidate)

    if not pool:
        raise RuntimeError("No valid app/flow/supi triples were found in the network state.")
    return pool


def _select_candidates(
    pool: Sequence[UserInputCandidate],
    *,
    total_count: int,
    seed: int,
) -> List[UserInputCandidate]:
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    if total_count >= len(shuffled):
        return shuffled
    return shuffled[:total_count]


def build_output(
    payload: Dict[str, Any],
    *,
    count: int,
    target_success_rate: float,
    seed: int,
) -> Dict[str, Any]:
    del target_success_rate

    if count <= 0:
        raise ValueError("count must be positive")

    pool = generate_candidate_pool(payload)
    selected = _select_candidates(pool, total_count=count, seed=seed)

    return {
        "meta": {
            "count": len(selected),
            "seed": seed,
            "source_pool_size": len(pool),
        },
        "records": [
            {
                "user_input": item.user_input,
                "messages": list(item.messages),
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
    parser = argparse.ArgumentParser(description="Generate minimal Chinese user inputs from current network state.")
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
        help="Reserved for backward compatibility. No longer affects sampling.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive.")

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
    print(f"Source pool size: {meta['source_pool_size']}")


if __name__ == "__main__":
    main()