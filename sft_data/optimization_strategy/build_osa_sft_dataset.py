from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain.collaboration import PlanningRequest
from domain.policy_plan import PolicyPlanDraft

from sft_data.builders import CanonicalBuilderConfig, build_canonical_dataset
from sft_data.common import ArtifactPair, processed_dir, rejects_dir
from sft_data.schemas import (
    BuildReport,
    DatasetMessage,
    SupervisedSftRecord,
    load_build_report,
    save_build_report,
    write_jsonl,
)


def _format_request_message(request: PlanningRequest) -> str:
    operation_intent = request.operation_intent.model_dump(mode="json")
    context = request.context.model_dump(mode="json")
    return (
        "Operation intent:\n"
        f"{json.dumps(operation_intent, ensure_ascii=False)}\n\n"
        "Collaboration context:\n"
        f"{json.dumps(context, ensure_ascii=False)}\n\n"
        "Fetch network status first, then run optimization and return a structured policy draft."
    )


def _validate_target(payload: Dict[str, Any]) -> Dict[str, Any]:
    draft = PolicyPlanDraft.model_validate(payload)
    if not draft.all_policies:
        raise ValueError("PolicyPlanDraft must contain at least one policy")
    for index, policy in enumerate(draft.all_policies, start=1):
        details = policy.policy_details
        if policy.policy_type == "SmPolicyDecision":
            pcc_rules = details.get("pccRules")
            qos_decs = details.get("qosDecs")
            if not isinstance(pcc_rules, dict) or not pcc_rules:
                raise ValueError(f"Policy #{index} is missing non-empty pccRules")
            if not isinstance(qos_decs, dict) or not qos_decs:
                raise ValueError(f"Policy #{index} is missing non-empty qosDecs")
        if policy.policy_type == "UrspRuleRequest":
            route_sets = details.get("routeSelParamSets")
            if not isinstance(route_sets, list) or not route_sets:
                raise ValueError(f"Policy #{index} is missing routeSelParamSets")
    return draft.model_dump(mode="json")


def _build_record(pair: ArtifactPair) -> SupervisedSftRecord:
    request_model = PlanningRequest.model_validate(pair.request.get("payload", {}))
    target = _validate_target(pair.response.get("payload", {}))
    return SupervisedSftRecord(
        sample_id=f"osa:{pair.response.get('artifact_id')}",
        task="osa_sft",
        agent="optimization_strategy",
        messages=[DatasetMessage(role="user", content=_format_request_message(request_model))],
        target=target,
        metadata={
            "matched_by": pair.matched_by,
            "request_artifact_id": pair.request.get("artifact_id"),
            "response_artifact_id": pair.response.get("artifact_id"),
            "correlation_id": pair.response.get("correlation_id"),
        },
    )


def build_osa_sft_records(project_root: Path) -> Tuple[List[SupervisedSftRecord], List[Dict[str, Any]], BuildReport]:
    records, rejects, artifact_total = build_canonical_dataset(
        CanonicalBuilderConfig(
            project_root=project_root,
            request_relative="runtime/interfaces/intent_encoding__optimization_strategy/requests",
            response_relative="runtime/interfaces/optimization_strategy__intent_encoding/responses",
            build_record=_build_record,
        )
    )
    report = BuildReport(
        artifact_total=artifact_total,
        paired_total=len(records),
        osa_sft_samples=len(records),
        reject_total=len(rejects),
    )
    return records, rejects, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the OSA supervised SFT dataset.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=processed_dir("optimization_strategy") / "osa_sft_v1.jsonl",
    )
    parser.add_argument(
        "--reject-output",
        type=Path,
        default=rejects_dir("optimization_strategy") / "osa_sft_rejects_v1.jsonl",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=processed_dir("optimization_strategy") / "build_report_v1.json",
    )
    args = parser.parse_args()

    records, rejects, report = build_osa_sft_records(args.project_root)
    write_jsonl(args.output, records)
    write_jsonl(args.reject_output, rejects)
    merged_report = load_build_report(args.report_output)
    merged_report.artifact_total = report.artifact_total
    merged_report.paired_total = report.paired_total
    merged_report.osa_sft_samples = report.osa_sft_samples
    merged_report.reject_total = report.reject_total
    save_build_report(args.report_output, merged_report)
    print(f"Wrote {len(records)} OSA SFT rows to {args.output}")
    print(f"Rejected {len(rejects)} rows")


if __name__ == "__main__":
    main()

