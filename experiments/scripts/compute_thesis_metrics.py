from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from experiments.scripts.common import RAW_RUN_ROOT, RESULTS_ROOT, SUMMARY_ROOT, load_json, write_json


QOS_POLICY_TYPE = "SmPolicyDecision"
MOBILITY_POLICY_TYPE = "AmPolicyData"
MOBILITY_POLICY_TYPES = frozenset({MOBILITY_POLICY_TYPE, "PcfAmPolicyControlPolicyAssociation"})


@dataclass(frozen=True)
class TargetIds:
    supis: frozenset[str]
    app_ids: frozenset[str]
    flow_ids: frozenset[str]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise TypeError(f"{path} contains a non-object JSONL row")
        rows.append(payload)
    return rows


def _normalize_dispatch_receipts(container: Mapping[str, Any]) -> List[Dict[str, Any]]:
    receipts: List[Dict[str, Any]] = []
    for feedback_name in ("qos_feedback", "mobility_feedback"):
        feedback = container.get(feedback_name)
        if not isinstance(feedback, dict):
            continue
        value = feedback.get("dispatch_receipts")
        if not isinstance(value, list):
            continue
        for receipt in value:
            if isinstance(receipt, dict):
                receipts.append(receipt)
    return receipts


def _receipt_applied(receipt: Mapping[str, Any]) -> bool:
    return (
        str(receipt.get("status") or "").strip().lower() == "success"
        and str(receipt.get("execution_status") or "").strip().upper() == "APPLIED"
        and str(receipt.get("compliance_status") or "").strip().upper() == "COMPLIANT"
    )


def _collect_ids_from_mapping(payload: Mapping[str, Any], *, supis: set[str], app_ids: set[str], flow_ids: set[str]) -> None:
    for key in ("supi", "ue_id"):
        value = str(payload.get(key) or "").strip()
        if value:
            supis.add(value)
    for key in ("app_id",):
        value = str(payload.get(key) or "").strip()
        if value:
            app_ids.add(value)
    for key in ("flow_id",):
        value = str(payload.get(key) or "").strip()
        if value:
            flow_ids.add(value)


def _collect_ids_from_sequence(items: Sequence[Any], *, supis: set[str], app_ids: set[str], flow_ids: set[str]) -> None:
    for item in items:
        if isinstance(item, Mapping):
            _collect_ids_from_mapping(item, supis=supis, app_ids=app_ids, flow_ids=flow_ids)


def _target_ids_from_trace(trace: Mapping[str, Any], *, include_execution_artifacts: bool = True) -> TargetIds:
    supis: set[str] = set()
    app_ids: set[str] = set()
    flow_ids: set[str] = set()

    operation_intent = trace.get("operation_intent")
    if isinstance(operation_intent, Mapping):
        _collect_ids_from_mapping(operation_intent, supis=supis, app_ids=app_ids, flow_ids=flow_ids)
        flows = operation_intent.get("flows")
        if isinstance(flows, list):
            _collect_ids_from_sequence(flows, supis=supis, app_ids=app_ids, flow_ids=flow_ids)
        envelopes = operation_intent.get("qos_target_envelopes")
        if isinstance(envelopes, list):
            _collect_ids_from_sequence(envelopes, supis=supis, app_ids=app_ids, flow_ids=flow_ids)
        grounding = operation_intent.get("grounding_evidence")
        if isinstance(grounding, Mapping):
            grounded_flows = grounding.get("grounded_flows")
            if isinstance(grounded_flows, list):
                _collect_ids_from_sequence(grounded_flows, supis=supis, app_ids=app_ids, flow_ids=flow_ids)
            grounded_apps = grounding.get("grounded_apps")
            if isinstance(grounded_apps, list):
                _collect_ids_from_sequence(grounded_apps, supis=supis, app_ids=app_ids, flow_ids=flow_ids)

    if include_execution_artifacts:
        unified_plan = trace.get("policy_plan")
        if isinstance(unified_plan, Mapping):
            for key in ("policy_drafts", "all_policies", "partial_policies", "approved_policies"):
                value = unified_plan.get(key)
                if isinstance(value, list):
                    _collect_ids_from_sequence(value, supis=supis, app_ids=app_ids, flow_ids=flow_ids)

        for receipt in _normalize_dispatch_receipts(trace):
            _collect_ids_from_mapping(receipt, supis=supis, app_ids=app_ids, flow_ids=flow_ids)
            upstream = receipt.get("upstream")
            if isinstance(upstream, Mapping):
                request_body = upstream.get("request_body")
                if isinstance(request_body, Mapping):
                    _collect_ids_from_mapping(request_body, supis=supis, app_ids=app_ids, flow_ids=flow_ids)
            policy_details = receipt.get("policy_details")
            if isinstance(policy_details, Mapping):
                request = policy_details.get("request")
                if isinstance(request, Mapping):
                    _collect_ids_from_mapping(request, supis=supis, app_ids=app_ids, flow_ids=flow_ids)
                upstream_context = policy_details.get("upstreamSmPolicyContextData")
                if isinstance(upstream_context, Mapping):
                    _collect_ids_from_mapping(upstream_context, supis=supis, app_ids=app_ids, flow_ids=flow_ids)

    return TargetIds(
        supis=frozenset(supis),
        app_ids=frozenset(app_ids),
        flow_ids=frozenset(flow_ids),
    )


def _domains_from_receipts(receipts: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    domains: set[str] = set()
    for receipt in receipts:
        policy_type = str(receipt.get("policy_type") or "").strip()
        if policy_type == QOS_POLICY_TYPE:
            domains.add("qos")
        elif policy_type in MOBILITY_POLICY_TYPES:
            domains.add("mobility")
    return frozenset(domains)


def _domains_from_unified_plan(plan: Mapping[str, Any]) -> frozenset[str]:
    domains: set[str] = set()
    approved_policies = plan.get("approved_policies")
    if isinstance(approved_policies, list):
        for item in approved_policies:
            if not isinstance(item, Mapping):
                continue
            value = str(item.get("domain") or "").strip().lower()
            if value:
                domains.add(value)
            policy_type = str(item.get("policy_type") or "").strip()
            if policy_type == QOS_POLICY_TYPE:
                domains.add("qos")
            elif policy_type in MOBILITY_POLICY_TYPES:
                domains.add("mobility")
    if not domains:
        execution_order = plan.get("execution_order")
        if isinstance(execution_order, list):
            for value in execution_order:
                text = str(value or "").strip().lower()
                if text:
                    domains.add(text)
    return frozenset(domains)


def _case_is_dispatch_success(item: Mapping[str, Any]) -> bool:
    receipts = _normalize_dispatch_receipts(item)
    return bool(receipts) and all(_receipt_applied(receipt) for receipt in receipts)


def _case_is_completed_correctly(item: Mapping[str, Any], method_slug: str = "") -> bool:
    return (
        item.get("status") == "success"
        and bool(item.get("completed"))
        and _case_is_dispatch_success(item)
        and _case_has_eic(item, method_slug=method_slug)
    )


def _round_is_success(trace: Mapping[str, Any]) -> bool:
    receipts = _normalize_dispatch_receipts(trace)
    return bool(receipts) and all(_receipt_applied(receipt) for receipt in receipts)


def _case_generated_policy(item: Mapping[str, Any]) -> bool:
    unified_plan = item.get("unified_plan")
    if not isinstance(unified_plan, Mapping):
        return False
    approved = unified_plan.get("approved_policies")
    return isinstance(approved, list) and len(approved) > 0


def _resolved_operation_intent(trace: Mapping[str, Any]) -> bool:
    operation_intent = trace.get("operation_intent")
    if not isinstance(operation_intent, Mapping):
        return False
    return str(operation_intent.get("resolution_status") or "").strip().lower() == "resolved"


def _downstream_ids(item: Mapping[str, Any]) -> TargetIds:
    unified_plan = item.get("unified_plan")
    if isinstance(unified_plan, Mapping):
        return _target_ids_from_trace(
            {
                "policy_plan": unified_plan,
                "qos_feedback": item.get("qos_feedback"),
                "mobility_feedback": item.get("mobility_feedback"),
            }
        )
    return TargetIds(frozenset(), frozenset(), frozenset())


def _ids_preserved(trace: Mapping[str, Any], item: Mapping[str, Any]) -> bool:
    trace_ids = _target_ids_from_trace(trace, include_execution_artifacts=False)
    downstream_ids = _downstream_ids(item)

    if trace_ids.flow_ids:
        return bool(downstream_ids.flow_ids) and downstream_ids.flow_ids.issubset(trace_ids.flow_ids)
    if trace_ids.app_ids:
        return bool(downstream_ids.app_ids) and downstream_ids.app_ids.issubset(trace_ids.app_ids)
    if trace_ids.supis:
        return bool(downstream_ids.supis) and downstream_ids.supis.issubset(trace_ids.supis)
    return False


def _operation_domains(trace: Mapping[str, Any]) -> frozenset[str]:
    operation_intent = trace.get("operation_intent")
    if not isinstance(operation_intent, Mapping):
        return frozenset()
    return frozenset(
        str(value or "").strip().lower()
        for value in (operation_intent.get("requested_domains") or [])
        if str(value or "").strip()
    )


def _approved_policy_domains(item: Mapping[str, Any]) -> frozenset[str]:
    unified_plan = item.get("unified_plan")
    if not isinstance(unified_plan, Mapping):
        return frozenset()
    return _domains_from_unified_plan(unified_plan)


def _receipt_domains(item: Mapping[str, Any]) -> frozenset[str]:
    return _domains_from_receipts(_normalize_dispatch_receipts(item))


def _qos_targets_present(trace: Mapping[str, Any]) -> bool:
    operation_intent = trace.get("operation_intent")
    if not isinstance(operation_intent, Mapping):
        return False
    flows = operation_intent.get("flows")
    if isinstance(flows, list) and any(isinstance(flow, Mapping) and str(flow.get("flow_id") or "").strip() for flow in flows):
        return True
    envelopes = operation_intent.get("qos_target_envelopes")
    if isinstance(envelopes, list) and any(isinstance(env, Mapping) and str(env.get("flow_id") or "").strip() for env in envelopes):
        return True
    return False


def _mobility_targets_present(trace: Mapping[str, Any]) -> bool:
    operation_intent = trace.get("operation_intent")
    if not isinstance(operation_intent, Mapping):
        return False
    mobility_intent = operation_intent.get("mobility_intent")
    if isinstance(mobility_intent, Mapping) and any(str(value or "").strip() for value in mobility_intent.values()):
        return True
    grounding = operation_intent.get("grounding_evidence")
    if not isinstance(grounding, Mapping):
        return False
    grounded_targets = grounding.get("grounded_mobility_targets")
    if not isinstance(grounded_targets, Mapping):
        return False
    summary = grounded_targets.get("summary")
    if isinstance(summary, Mapping) and any(value not in (None, "", [], {}) for value in summary.values()):
        return True
    candidates = grounded_targets.get("candidates")
    return isinstance(candidates, list) and any(isinstance(candidate, Mapping) and candidate for candidate in candidates)


def _goal_targets_preserved(trace: Mapping[str, Any], item: Mapping[str, Any]) -> bool:
    op_domains = _operation_domains(trace)
    approved_domains = _approved_policy_domains(item)
    receipt_domains = _receipt_domains(item)

    if "qos" in op_domains:
        if not _qos_targets_present(trace):
            return False
        if "qos" not in approved_domains or "qos" not in receipt_domains:
            return False

    if "mobility" in op_domains:
        if not _mobility_targets_present(trace):
            return False
        if "mobility" not in approved_domains or "mobility" not in receipt_domains:
            return False

    return True


def _case_has_semantic_binding_consistency(item: Mapping[str, Any]) -> bool:
    round_traces = item.get("round_traces")
    if not isinstance(round_traces, list) or not round_traces:
        return False
    final_trace = round_traces[-1]
    if not isinstance(final_trace, Mapping):
        return False

    global_intent = item.get("global_intent")
    if not isinstance(global_intent, Mapping):
        return False
    requested_domains = {
        str(value or "").strip().lower()
        for value in (global_intent.get("requested_domains") or [])
        if str(value or "").strip()
    }
    if not requested_domains:
        return False

    unified_plan = item.get("unified_plan")
    if not isinstance(unified_plan, Mapping):
        return False
    operation_domains = _operation_domains(final_trace)
    plan_domains = _approved_policy_domains(item)
    receipt_domains = _receipt_domains(item)
    if not operation_domains:
        return False
    if requested_domains != set(operation_domains):
        return False
    if requested_domains != set(plan_domains):
        return False
    if requested_domains != set(receipt_domains):
        return False

    if not _resolved_operation_intent(final_trace):
        return False

    if not _ids_preserved(final_trace, item):
        return False

    return _goal_targets_preserved(final_trace, item)


def _is_single_agent_method(method_slug: str) -> bool:
    return str(method_slug or "").strip().lower() in {"b1", "b2", "b3"}


def _operation_intent_reached_single_agent_final(trace: Mapping[str, Any]) -> bool:
    operation_intent = trace.get("operation_intent")
    if not isinstance(operation_intent, Mapping):
        return False
    return str(operation_intent.get("resolution_status") or "").strip().lower() in {
        "resolved",
        "final_product",
    }


def _case_has_single_agent_eic(item: Mapping[str, Any]) -> bool:
    round_traces = item.get("round_traces")
    if not isinstance(round_traces, list) or not round_traces:
        return False
    final_trace = round_traces[-1]
    if not isinstance(final_trace, Mapping):
        return False

    global_intent = item.get("global_intent")
    if not isinstance(global_intent, Mapping):
        return False
    requested_domains = {
        str(value or "").strip().lower()
        for value in (global_intent.get("requested_domains") or [])
        if str(value or "").strip()
    }
    if not requested_domains:
        requested_domains = set(_operation_domains(final_trace))
    if not requested_domains:
        return False

    unified_plan = item.get("unified_plan")
    if not isinstance(unified_plan, Mapping):
        return False

    operation_domains = _operation_domains(final_trace)
    plan_domains = _approved_policy_domains(item)
    receipt_domains = _receipt_domains(item)
    if not operation_domains:
        return False
    if requested_domains != set(operation_domains):
        return False
    if requested_domains != set(plan_domains):
        return False
    if requested_domains != set(receipt_domains):
        return False

    if not _operation_intent_reached_single_agent_final(final_trace):
        return False

    if not _ids_preserved(final_trace, item):
        return False

    return _goal_targets_preserved(final_trace, item)


def _case_has_eic(item: Mapping[str, Any], method_slug: str = "") -> bool:
    if _is_single_agent_method(method_slug):
        return _case_has_single_agent_eic(item)
    return _case_has_semantic_binding_consistency(item)


def _first_round_failed(item: Mapping[str, Any]) -> bool:
    round_traces = item.get("round_traces")
    if not isinstance(round_traces, list) or not round_traces:
        return True
    first_trace = round_traces[0]
    if not isinstance(first_trace, Mapping):
        return True
    return not _round_is_success(first_trace)


def _mean(values: Iterable[int]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(float(statistics.mean(values)), 4)


def _mean_numeric(values: Iterable[Any]) -> float:
    numeric_values: List[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            numeric_values.append(float(value))
            continue
        if isinstance(value, str) and value.strip():
            try:
                numeric_values.append(float(value))
            except ValueError:
                continue
    if not numeric_values:
        return 0.0
    return round(float(statistics.mean(numeric_values)), 4)


def _resolve_run_records_path(summary_path: Path, summary_payload: Mapping[str, Any], raw_dir: Path) -> Path:
    artifacts = summary_payload.get("artifacts")
    candidate_text = ""
    if isinstance(artifacts, Mapping):
        candidate_text = str(artifacts.get("run_records") or "").strip()
    if candidate_text:
        candidate = Path(candidate_text)
        if candidate.exists():
            return candidate
        candidate = raw_dir / candidate.name
        if candidate.exists():
            return candidate

    summary_name = summary_path.name
    if "_summary_" not in summary_name:
        raise FileNotFoundError(f"Cannot infer raw run file from summary name: {summary_name}")
    inferred_name = summary_name.replace("_summary_", "_runs_").replace(".json", ".jsonl")
    candidate = raw_dir / inferred_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve raw run file for summary {summary_path}")


def _method_slug_from_summary_name(summary_path: Path) -> str:
    name = summary_path.name
    if "_summary_" not in name:
        return summary_path.stem
    return name.split("_summary_", 1)[0]


def compute_metrics_for_run(summary_path: Path, raw_path: Path) -> Dict[str, Any]:
    summary_payload = load_json(summary_path)
    rows = _read_jsonl(raw_path)
    if not rows:
        raise RuntimeError(f"No run records found in {raw_path}")

    total = len(rows)
    summary_total = summary_payload.get("total_cases")
    if isinstance(summary_total, int) and summary_total != total:
        raise ValueError(f"Summary total_cases={summary_total} does not match raw rows={total} for {summary_path.name}")

    method_slug = _method_slug_from_summary_name(summary_path)
    eic_count = sum(1 for row in rows if _case_has_eic(row, method_slug=method_slug))
    pgr_count = sum(1 for row in rows if _case_generated_policy(row))
    dsr_count = sum(1 for row in rows if _case_is_dispatch_success(row))
    tcr_count = sum(1 for row in rows if _case_is_completed_correctly(row, method_slug=method_slug))
    first_round_failed_count = sum(1 for row in rows if _first_round_failed(row))
    recovered_count = sum(
        1 for row in rows if _first_round_failed(row) and _case_is_completed_correctly(row, method_slug=method_slug)
    )

    scenario_ids = sorted({str(row.get("scenario_id") or "").strip() for row in rows if str(row.get("scenario_id") or "").strip()})

    return {
        "summary_file": str(summary_path),
        "raw_run_file": str(raw_path),
        "method_slug": method_slug,
        "scenario_ids": scenario_ids,
        "total_cases": total,
        "eic_count": eic_count,
        "eic_rate": round(eic_count / total, 4) if total else 0.0,
        "pgr_count": pgr_count,
        "pgr_rate": round(pgr_count / total, 4) if total else 0.0,
        "dsr_count": dsr_count,
        "dsr_rate": round(dsr_count / total, 4) if total else 0.0,
        "crr_recovered_count": recovered_count,
        "crr_first_round_failed_count": first_round_failed_count,
        "crr_rate": round(recovered_count / first_round_failed_count, 4) if first_round_failed_count else 0.0,
        "tcr_count": tcr_count,
        "tcr_rate": round(tcr_count / total, 4) if total else 0.0,
        "avg_round_count": _mean(int(row.get("round_count") or 0) for row in rows),
        "avg_retry_count": _mean(int(row.get("retry_count") or 0) for row in rows),
        "avg_elapsed_ms": _mean_numeric(row.get("elapsed_ms") for row in rows),
        "legacy_summary": {
            "correct_case_rate": summary_payload.get("correct_case_rate"),
            "completed_case_rate": summary_payload.get("completed_case_rate"),
            "main_judgement_wrong_count": summary_payload.get("main_judgement_wrong_count"),
        },
    }


def _aggregate_runs(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total_cases = sum(int(record["total_cases"]) for record in records)
    eic_count = sum(int(record["eic_count"]) for record in records)
    pgr_count = sum(int(record["pgr_count"]) for record in records)
    dsr_count = sum(int(record["dsr_count"]) for record in records)
    tcr_count = sum(int(record["tcr_count"]) for record in records)
    first_round_failed_count = sum(int(record["crr_first_round_failed_count"]) for record in records)
    recovered_count = sum(int(record["crr_recovered_count"]) for record in records)
    weighted_rounds = sum(float(record["avg_round_count"]) * int(record["total_cases"]) for record in records)
    weighted_retries = sum(float(record["avg_retry_count"]) * int(record["total_cases"]) for record in records)
    weighted_elapsed_ms = sum(float(record.get("avg_elapsed_ms") or 0.0) * int(record["total_cases"]) for record in records)

    return {
        "run_count": len(records),
        "total_cases": total_cases,
        "eic_rate": round(eic_count / total_cases, 4) if total_cases else 0.0,
        "pgr_rate": round(pgr_count / total_cases, 4) if total_cases else 0.0,
        "dsr_rate": round(dsr_count / total_cases, 4) if total_cases else 0.0,
        "crr_rate": round(recovered_count / first_round_failed_count, 4) if first_round_failed_count else 0.0,
        "tcr_rate": round(tcr_count / total_cases, 4) if total_cases else 0.0,
        "avg_round_count": round(weighted_rounds / total_cases, 4) if total_cases else 0.0,
        "avg_retry_count": round(weighted_retries / total_cases, 4) if total_cases else 0.0,
        "avg_elapsed_ms": round(weighted_elapsed_ms / total_cases, 4) if total_cases else 0.0,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write CSV without rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method_slug",
        "scenario_ids",
        "total_cases",
        "eic_count",
        "eic_rate",
        "pgr_count",
        "pgr_rate",
        "dsr_count",
        "dsr_rate",
        "crr_recovered_count",
        "crr_first_round_failed_count",
        "crr_rate",
        "tcr_count",
        "tcr_rate",
        "avg_round_count",
        "avg_retry_count",
        "avg_elapsed_ms",
        "summary_file",
        "raw_run_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key) for key in fieldnames}
            payload["scenario_ids"] = ",".join(row.get("scenario_ids") or [])
            writer.writerow(payload)


def _resolve_summary_paths(summary_dir: Path, patterns: Sequence[str]) -> List[Path]:
    matched_paths: List[Path] = []
    seen_paths: set[Path] = set()
    for pattern in patterns:
        for summary_path in sorted(summary_dir.glob(pattern)):
            if summary_path in seen_paths:
                continue
            matched_paths.append(summary_path)
            seen_paths.add(summary_path)
    return matched_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute thesis metrics (EIC/PGR/DSR/CRR/TCR + average rounds/retries) from experiment summaries and raw run traces.",
    )
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_ROOT)
    parser.add_argument("--raw-dir", type=Path, default=RAW_RUN_ROOT)
    parser.add_argument(
        "--summary-glob",
        dest="summary_globs",
        action="append",
        metavar="PATTERN",
        help="Repeat this option to include multiple summary file patterns.",
    )
    parser.add_argument("--output-json", type=Path, default=RESULTS_ROOT / "thesis_metrics.json")
    parser.add_argument("--output-csv", type=Path, default=RESULTS_ROOT / "thesis_metrics.csv")
    args = parser.parse_args(argv)
    args.summary_globs = args.summary_globs or ["*.json"]
    return args


def main() -> None:
    args = parse_args()
    summary_paths = _resolve_summary_paths(args.summary_dir, args.summary_globs)
    if not summary_paths:
        raise FileNotFoundError(f"No summary files matched {args.summary_globs} under {args.summary_dir}")

    run_metrics: List[Dict[str, Any]] = []
    for summary_path in summary_paths:
        summary_payload = load_json(summary_path)
        raw_path = _resolve_run_records_path(summary_path, summary_payload, args.raw_dir)
        run_metrics.append(compute_metrics_for_run(summary_path, raw_path))

    payload = {
        "runs": run_metrics,
        "overall": _aggregate_runs(run_metrics),
    }
    write_json(args.output_json, payload)
    _write_csv(args.output_csv, run_metrics)

    print(f"Computed thesis metrics for {len(run_metrics)} run summaries")
    print(f"JSON -> {args.output_json}")
    print(f"CSV -> {args.output_csv}")


if __name__ == "__main__":
    main()
