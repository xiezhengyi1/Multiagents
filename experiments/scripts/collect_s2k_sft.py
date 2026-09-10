"""Collect only evidence-complete, topology-bound S2K episodes for SFT."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from experiments.contracts import ExperimentCase, ExperimentResult
from training.artifacts import read_jsonl, write_jsonl
from training.projection import project_sft_examples
from training.records import EpisodeRecords


def _events(records: EpisodeRecords, agent: str, message_type: str) -> list[dict[str, Any]]:
    return [
        event
        for event in records.events
        if event["source_agent"] == agent and event["message_type"] == message_type
    ]


def _payloads(records: EpisodeRecords, agent: str, message_type: str) -> list[dict[str, Any]]:
    return [dict(event["payload"]) for event in _events(records, agent, message_type)]


def _has_value(payloads: Iterable[dict[str, Any]], key: str) -> bool:
    return any(payload.get(key) not in (None, [], {}, "") for payload in payloads)


def _required_evidence(records: EpisodeRecords, contract: list[str]) -> list[str]:
    observations = _payloads(records, "simulator_adapter", "observation")
    main = _payloads(records, "main", "proposal")
    sba = _payloads(records, "sba", "proposal")
    handoffs = _payloads(records, "sba_compiler", "handoff")
    ppa = _payloads(records, "ppa", "proposal")
    available = {
        "snapshot_id": _has_value(observations, "snapshot_id"),
        "flow_catalog": _has_value(handoffs, "flow_catalog"),
        "pdu_session_pool": _has_value(handoffs, "pdu_session_pool"),
        "subscription": _has_value(handoffs, "subscription"),
        "ue_policy_association": _has_value(handoffs, "ue_policy_associations"),
        "am_policy_association": _has_value(ppa, "am_policy"),
        "qos_binding": _has_value(sba, "qos_operation_constraints"),
        "resource_snapshot": any(
            payload.get("entities") not in (None, [], {}) and payload.get("metrics") not in (None, [], {})
            for payload in observations
        ),
        "user_constraints": _has_value(main, "control_semantics"),
    }
    return [name for name in contract if not available[name]]


def _operations(records: EpisodeRecords) -> list[dict[str, Any]]:
    handoffs = _payloads(records, "ppa_compiler", "handoff")
    if len(handoffs) != 1:
        return []
    return list(handoffs[0]["operations"])


def _verified_execution(records: EpisodeRecords) -> bool:
    feedback = _payloads(records, "execution_dispatcher", "execution_feedback")
    if len(feedback) != 1 or feedback[0]["status"] != "success":
        return False
    receipts = feedback[0]["receipts"]
    return bool(receipts) and all(
        receipt["status"] == "success" and receipt["result"]["status"] == "verified"
        for receipt in receipts
    )


def _missing_acceptance(records: EpisodeRecords, checks: list[str], resolution: str) -> list[str]:
    operations = _operations(records)
    if resolution == "infeasible":
        ppa = _payloads(records, "ppa", "proposal")
        plan = ppa[-1] if ppa else {}
        available = {
            "partial_plan": plan.get("planning_status") == "partial_plan",
            "no_action": not operations,
            "no_open_questions": plan.get("open_questions") == [] and plan.get("upstream_requests") == [],
        }
    else:
        verified = _verified_execution(records)
        domains = {operation["domain"] for operation in operations}
        resources = {operation["resource_type"] for operation in operations}
        available = {
            "receipt_success": verified,
            "qos_changed": verified and "qos" in domains,
            "route_unchanged": verified and "UePolicySet" not in resources,
            "target_flow_route_changed": verified and "UePolicySet" in resources,
            "no_dynamic_pdu": "PduSession" not in resources,
            "am_policy_version_advanced": verified and "mobility" in domains,
            "stage_order_preserved": verified and len(operations) >= 2,
            "safety_budget_preserved": verified and "qos" in domains,
        }
    return [name for name in checks if not available[name]]


def audit_result(case: ExperimentCase, raw_result: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = ExperimentResult.model_validate(
        {key: value for key, value in raw_result.items() if key != "baseline_id"}
    )
    expected = dict(case.metadata["expected_terminal"])
    audit: dict[str, Any] = {
        "case_id": case.case_id,
        "episode_id": result.episode_id,
        "scenario_id": case.metadata["scenario_id"],
        "config_hash": case.metadata["scenario_config_sha256"],
        "terminal": result.terminal.model_dump(mode="json"),
        "expected_terminal": expected,
        "baseline_id": raw_result.get("baseline_id"),
        "sft_disposition": "diagnostic",
    }
    if raw_result.get("baseline_id") != case.case_id:
        audit["reason"] = "missing_clean_baseline"
        return audit, []
    actual = result.terminal.model_dump(mode="json")
    if any(actual[name] != expected[name] for name in ("outcome", "resolution", "reason_code")):
        audit["reason"] = "terminal_mismatch"
        return audit, []
    records = EpisodeRecords.load(Path(result.episode_directory))
    missing_evidence = _required_evidence(records, list(case.metadata["evidence_contract"]))
    if missing_evidence:
        audit["reason"] = "evidence_missing"
        audit["missing_evidence"] = missing_evidence
        return audit, []
    missing_checks = _missing_acceptance(
        records,
        list(case.metadata["acceptance_checks"]),
        result.terminal.resolution,
    )
    if missing_checks:
        audit["reason"] = "acceptance_missing"
        audit["missing_acceptance_checks"] = missing_checks
        return audit, []
    examples = [example.model_dump(mode="json") for example in project_sft_examples(records)]
    if not examples:
        audit["reason"] = "no_accepted_model_trace"
        return audit, []
    audit["sft_disposition"] = "accepted"
    audit["reason"] = "gold_terminal_and_evidence_verified"
    audit["accepted_example_ids"] = [example["example_id"] for example in examples]
    return audit, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--status", type=Path, action="append", default=[])
    parser.add_argument("--sft-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    cases = {case.case_id: case for case in (ExperimentCase.model_validate(row) for row in read_jsonl(args.cases))}
    audits: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    result_case_ids: set[str] = set()
    for path in args.results:
        for result in read_jsonl(path):
            case = cases[result["case_id"]]
            audit, accepted = audit_result(case, result)
            audits.append(audit)
            examples.extend(accepted)
            result_case_ids.add(case.case_id)
    for path in args.status:
        for status in read_jsonl(path):
            case_id = status["case_id"]
            if case_id not in result_case_ids:
                audits.append(
                    {
                        "case_id": case_id,
                        "scenario_id": cases[case_id].metadata["scenario_id"],
                        "sft_disposition": "diagnostic",
                        "reason": "no_result",
                        "status": status,
                    }
                )
    write_jsonl(args.audit_output, audits)
    write_jsonl(args.sft_output, examples)
    print(f"accepted_cases={sum(row['sft_disposition'] == 'accepted' for row in audits)} sft_examples={len(examples)}")


if __name__ == "__main__":
    main()
