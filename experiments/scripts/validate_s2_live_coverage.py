from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from control_runtime.integrations.testbed import TestbedSimulatorAdapter
from control_runtime.simulation import ObservationRequest
from experiments.contracts import ExperimentCase
from training.artifacts import read_jsonl

SUPI_PATTERN = re.compile(r"imsi-\d{5,15}")


def _tokens(value: object) -> frozenset[str]:
    return frozenset(re.findall(r"[0-9a-z]+", str(value).casefold()))


def _strings(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def _expected_references(task: dict[str, Any]) -> tuple[list[str], list[str]]:
    flows: list[str] = []
    apps: list[str] = []
    for key, value in task["expected_objects"].items():
        if "flow" in key:
            flows.extend(_strings(value))
        elif key == "app" or key.endswith("_app"):
            apps.extend(_strings(value))
    return flows, apps


def _single_token_match(name: str, candidates: dict[str, str]) -> str:
    matches = [candidate_id for candidate_id, value in candidates.items() if _tokens(name) == _tokens(value)]
    if len(matches) != 1:
        raise ValueError(f"{name!r} resolved to {matches}")
    return matches[0]


def validate_live_coverage(cases: list[ExperimentCase]) -> dict[str, Any]:
    adapter = TestbedSimulatorAdapter.from_env()
    projected = adapter.capture_snapshot()
    native = adapter._native()
    flow_names = {flow["id"]: flow["name"] for flow in native["flows"]}
    app_names = {flow["app_id"]: flow["app_name"] for flow in native["flows"]}
    flow_by_id = {flow["id"]: flow for flow in native["flows"]}
    reports = []
    optimizer_blocked_cases = []

    for case in cases:
        task = case.metadata["task"]
        expected_flows, expected_apps = _expected_references(task)
        matched_flow_ids = [
            _single_token_match(flow_name, flow_names) for flow_name in expected_flows
        ]
        matched_app_ids = [
            _single_token_match(app_name, app_names) for app_name in expected_apps
        ]
        supis = set(SUPI_PATTERN.findall(case.user_input or ""))
        supis.update(flow_by_id[flow_id]["supi"] for flow_id in matched_flow_ids)

        searches = []
        for expected_name, flow_id in zip(expected_flows, matched_flow_ids, strict=True):
            supi = flow_by_id[flow_id]["supi"]
            result = adapter.observe(
                ObservationRequest(
                    query_name="search_sm_flow_targets",
                    arguments={"supi": supi, "query": expected_name},
                )
            ).result
            matched_ids = [match["id"] for match in result["matches"]]
            if flow_id not in matched_ids:
                raise ValueError(f"{case.case_id} search missed {flow_id}")
            searches.append({"query": expected_name, "flow_id": flow_id, "supi": supi})

        subscriptions = {}
        for supi in sorted(supis):
            subscriptions[supi] = adapter.observe(
                ObservationRequest(
                    query_name="get_ue_slice_subscription",
                    arguments={"supi": supi},
                )
            ).result
            adapter.observe(
                ObservationRequest(
                    query_name="get_am_policy_context",
                    arguments={"supi": supi},
                )
            )

        flow_ids_by_supi: dict[str, list[str]] = defaultdict(list)
        for flow_id in matched_flow_ids:
            flow_ids_by_supi[flow_by_id[flow_id]["supi"]].append(flow_id)
        optimizer_previews = []
        for supi, flow_ids in sorted(flow_ids_by_supi.items()):
            preview = adapter.observe(
                ObservationRequest(
                    query_name="preview_qos_optimizer",
                    arguments={"supi": supi, "flow_ids": flow_ids},
                )
            ).result
            optimizer_previews.append(
                {"supi": supi, "flow_ids": flow_ids, "ready": preview["ready"]}
            )
            if not preview["ready"]:
                optimizer_blocked_cases.append(case.case_id)

        reports.append(
            {
                "case_id": case.case_id,
                "task_id": task["task_id"],
                "matched_flow_ids": matched_flow_ids,
                "matched_app_ids": matched_app_ids,
                "searches": searches,
                "subscription_supis": sorted(subscriptions),
                "optimizer_previews": optimizer_previews,
            }
        )

    return {
        "snapshot_id": projected.snapshot_id,
        "case_count": len(cases),
        "flow_count": len(native["flows"]),
        "session_count": len(native["sessions"]),
        "ran_node_count": len(native["ran_nodes"]),
        "core_node_count": len(native["core_nodes"]),
        "optimizer_blocked_cases": sorted(set(optimizer_blocked_cases)),
        "cases": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = [ExperimentCase.model_validate(row) for row in read_jsonl(args.cases)]
    report = validate_live_coverage(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["validate_live_coverage"]
