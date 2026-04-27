from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_user_inputs import (
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SEED,
    DEFAULT_TARGET_SUCCESS_RATE,
    build_output,
    iter_flow_anchors,
    load_network_state,
)
from sft_data.collect_workflow_trajectories import load_user_input_records
from workflows.main_control_orchestrator import MainControlOrchestrator, _parse_args as parse_workflow_cli_args


@dataclass(frozen=True)
class ExperimentRecord:
    user_input: str
    messages: List[Dict[str, str]]
    context: str
    scenario_id: str
    scenario_tags: List[str]


def _single_turn_messages(user_input: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": user_input}]


def _flow_brief(anchor: Mapping[str, Any]) -> str:
    app_name = str(anchor.get("app_name") or "").strip()
    flow_name = str(anchor.get("flow_name") or "").strip()
    return f"{app_name}/{flow_name}".strip("/")


def _sm_policy_templates(anchor: Mapping[str, Any]) -> List[str]:
    supi = str(anchor.get("supi") or "").strip()
    flow_brief = _flow_brief(anchor)
    return [
        f"请只调整 {supi} 的 SM policy，重点处理 {flow_brief} 的 QoS。",
        f"帮我看一下 {supi} 这条 {flow_brief}，只改 SM policy，不要动 AM policy。",
        f"请优化 {supi} 的 sm policy，优先处理 {flow_brief} 的带宽、时延和 5QI。",
        f"只做 qos 侧策略调整：{supi} 的 {flow_brief} 需要重算 sm policy。",
        f"please update SM policy only for {supi} on {flow_brief}, keep AM policy unchanged.",
    ]


def _am_policy_templates(anchor: Mapping[str, Any]) -> List[str]:
    supi = str(anchor.get("supi") or "").strip()
    flow_brief = _flow_brief(anchor)
    return [
        f"请只调整 {supi} 的 AM policy，重点检查 mobility、allowed NSSAI 和 RFSP，不要改 SM policy。",
        f"帮我处理 {supi} 的接入与移动性策略，只看 AM policy 和 service area restriction，业务是 {flow_brief}。",
        f"{supi} 这边只做 am policy 优化，关注 target NSSAI、access type 和 mobility 风险。",
        f"只修 mobility 侧：检查 {supi} 的 AM policy、RFSP、service area，不要碰 QoS policy。",
        f"please revise AM policy only for {supi}, focus on mobility, allowed NSSAI, target NSSAI, and RFSP.",
    ]


def _joint_policy_templates(anchor: Mapping[str, Any]) -> List[str]:
    supi = str(anchor.get("supi") or "").strip()
    flow_brief = _flow_brief(anchor)
    return [
        f"请联合优化 {supi} 的 SM policy 和 AM policy，业务是 {flow_brief}。",
        f"帮我一起重算 {supi} 的 qos 策略和 mobility 策略，别让 {flow_brief} 的切片选择冲突。",
        f"{supi} 这条 {flow_brief} 需要联合处理 sm policy 与 am policy，重点看 QoS 和 allowed NSSAI 的一致性。",
        f"please optimize both SM policy and AM policy for {supi} on {flow_brief}.",
    ]


def _deduplicate_experiment_records(records: Iterable[ExperimentRecord]) -> List[ExperimentRecord]:
    deduped: List[ExperimentRecord] = []
    seen: set[str] = set()
    for record in records:
        normalized = record.user_input.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(record)
    return deduped


def build_policy_experiment_records(payload: Mapping[str, Any], *, count: int) -> List[ExperimentRecord]:
    if count <= 0:
        raise ValueError("count must be positive")

    generated: List[ExperimentRecord] = []
    scenario_index = 1
    for anchor in iter_flow_anchors(dict(payload)):
        template_groups = [
            ("sm_policy", _sm_policy_templates(anchor)),
            ("am_policy", _am_policy_templates(anchor)),
            ("joint_policy", _joint_policy_templates(anchor)),
        ]
        for policy_tag, templates in template_groups:
            for template in templates:
                generated.append(
                    ExperimentRecord(
                        user_input=template,
                        messages=_single_turn_messages(template),
                        context="",
                        scenario_id=f"workflow-experiment-{scenario_index:05d}",
                        scenario_tags=["workflow_experiment", policy_tag],
                    )
                )
                scenario_index += 1

    deduped = _deduplicate_experiment_records(generated)
    if not deduped:
        raise RuntimeError("No experiment records were generated from the current network state.")
    return deduped[:count]


def _field_presence(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        for key in record.keys():
            counter[str(key)] += 1
    return dict(sorted(counter.items()))


def summarize_user_input_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    role_counter: Counter[str] = Counter()
    message_count_histogram: Counter[int] = Counter()
    scenario_tag_counter: Counter[str] = Counter()
    for record in records:
        messages = record.get("messages") or []
        message_count_histogram[len(messages)] += 1
        for message in messages:
            role_counter[str(message.get("role") or "").strip().lower() or "unknown"] += 1
        for tag in record.get("scenario_tags") or []:
            scenario_tag_counter[str(tag).strip()] += 1

    example_record = records[0] if records else {}
    return {
        "record_count": len(records),
        "field_presence": _field_presence(records),
        "message_roles": dict(sorted(role_counter.items())),
        "message_count_histogram": dict(sorted(message_count_histogram.items())),
        "scenario_tag_breakdown": dict(sorted(scenario_tag_counter.items())),
        "normalized_schema": {
            "record_index": "int",
            "user_input": "str",
            "messages": [{"role": "str", "content": "str"}],
            "context": "str",
            "scenario_id": "str",
            "scenario_tags": ["str"],
        },
        "example": {
            "user_input": example_record.get("user_input", ""),
            "messages": example_record.get("messages", []),
            "context": example_record.get("context", ""),
            "scenario_id": example_record.get("scenario_id", ""),
            "scenario_tags": example_record.get("scenario_tags", []),
        },
    }


def summarize_workflow_contract() -> Dict[str, Any]:
    cli_defaults = vars(parse_workflow_cli_args([]))
    run_signature = MainControlOrchestrator.run.__annotations__.copy()
    run_signature["return"] = str(run_signature.get("return", ""))
    return {
        "entry_file": str(PROJECT_ROOT / "workflows" / "main_control_orchestrator.py"),
        "entry_class": "MainControlOrchestrator",
        "entry_method": "run",
        "run_signature": {
            "user_input": "str",
            "scenario_id": "str",
            "scenario_tags": "list[str] | None",
            "return_type": run_signature["return"],
        },
        "cli_defaults": cli_defaults,
        "result_core_fields": [
            "session_id",
            "snapshot_id",
            "completed",
            "global_intent",
            "unified_plan",
            "qos_feedback",
            "mobility_feedback",
            "diagnosis",
            "round_count",
            "retry_count",
            "round_traces",
        ],
    }


def build_experiment_manifest(
    *,
    records: List[Dict[str, Any]],
    generated_records_path: Path,
    source_user_inputs_path: Optional[Path],
    network_input_path: Optional[Path],
    count: int,
    seed: int,
) -> Dict[str, Any]:
    return {
        "workflow_summary": summarize_workflow_contract(),
        "input_summary": summarize_user_input_records(records),
        "generation_config": {
            "source_user_inputs_path": str(source_user_inputs_path) if source_user_inputs_path else "",
            "network_input_path": str(network_input_path) if network_input_path else "",
            "count": count,
            "seed": seed,
            "generated_records_path": str(generated_records_path),
        },
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize workflow/input contract and prepare a batch of workflow experiment user inputs.",
    )
    parser.add_argument("--user-inputs", type=Path, default=None, help="Existing user-input JSON/JSONL to normalize and summarize.")
    parser.add_argument("--network-input", type=Path, default=None, help="Network-state JSON used when generating fresh records.")
    parser.add_argument("--count", type=int, default=DEFAULT_SAMPLE_COUNT, help="How many normalized records to prepare.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic sampling seed.")
    parser.add_argument(
        "--target-success-rate",
        type=float,
        default=DEFAULT_TARGET_SUCCESS_RATE,
        help="Passed through to the existing generator for compatibility. It does not affect sampling today.",
    )
    parser.add_argument(
        "--records-output",
        type=Path,
        default=PROJECT_ROOT / "workflow_experiment_user_inputs.json",
        help="Normalized experiment records output.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=PROJECT_ROOT / "workflow_experiment_manifest.json",
        help="Workflow/input summary output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")

    if args.user_inputs is not None:
        records = load_user_input_records(
            user_inputs_path=args.user_inputs,
            network_input_path=None,
            count=args.count,
            target_success_rate=args.target_success_rate,
            seed=args.seed,
        )
    else:
        network_state = load_network_state(args.network_input)
        baseline_payload = build_output(
            network_state,
            count=args.count,
            target_success_rate=args.target_success_rate,
            seed=args.seed,
        )
        baseline_records = baseline_payload.get("records", [])
        policy_records = build_policy_experiment_records(network_state, count=args.count)
        records = []
        for index, item in enumerate(policy_records, start=1):
            records.append(
                {
                    "record_index": index,
                    "user_input": item.user_input,
                    "messages": item.messages,
                    "context": item.context,
                    "scenario_id": item.scenario_id,
                    "scenario_tags": item.scenario_tags,
                }
            )
        generated_payload = {
            "meta": {
                "count": len(records),
                "seed": args.seed,
                "source_pool_size": len(policy_records),
                "baseline_generator_count": len(baseline_records) if isinstance(baseline_records, list) else 0,
                "policy_mix": {
                    "sm_policy": sum(1 for item in records if "sm_policy" in item["scenario_tags"]),
                    "am_policy": sum(1 for item in records if "am_policy" in item["scenario_tags"]),
                    "joint_policy": sum(1 for item in records if "joint_policy" in item["scenario_tags"]),
                },
            },
            "records": [
                {
                    "user_input": record["user_input"],
                    "messages": record["messages"],
                    "context": record["context"],
                    "scenario_id": record["scenario_id"],
                    "scenario_tags": record["scenario_tags"],
                }
                for record in records
            ],
        }
        _write_json(args.records_output, generated_payload)

    if args.user_inputs is not None:
        normalized_payload = {
            "meta": {
                "count": len(records),
                "seed": args.seed,
                "source": str(args.user_inputs),
            },
            "records": [
                {
                    "user_input": record["user_input"],
                    "messages": record["messages"],
                    "context": record.get("context", ""),
                    "scenario_id": record.get("scenario_id", ""),
                    "scenario_tags": record.get("scenario_tags", []),
                }
                for record in records
            ],
        }
        _write_json(args.records_output, normalized_payload)

    manifest = build_experiment_manifest(
        records=records,
        generated_records_path=args.records_output,
        source_user_inputs_path=args.user_inputs,
        network_input_path=args.network_input,
        count=args.count,
        seed=args.seed,
    )
    _write_json(args.manifest_output, manifest)

    print(f"Prepared {len(records)} experiment records -> {args.records_output}")
    print(f"Workflow/input summary -> {args.manifest_output}")


if __name__ == "__main__":
    main()
