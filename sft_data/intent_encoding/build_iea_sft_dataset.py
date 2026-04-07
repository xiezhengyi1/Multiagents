from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain.policy_plan import OperationIntent

from sft_data.builders import CanonicalBuilderConfig, build_canonical_dataset
from sft_data.common import ArtifactPair, dataset_dir, dataset_output_path
from sft_data.schemas import (
    BuildReport,
    DatasetMessage,
    SupervisedSftRecord,
    load_build_report,
    save_build_report,
    write_jsonl,
)


def _format_request_message(request: Dict[str, Any]) -> str:
    payload = request.get("payload", {})
    user_input = str(payload.get("user_input") or "").strip()
    if not user_input:
        raise ValueError("OperationIntentRequest.payload.user_input must not be empty")
    context = str(payload.get("context") or "N/A")
    session_id = str(request.get("session_id") or "")
    snapshot_id = str(request.get("snapshot_id") or "")
    return (
        f"User input:\n{user_input}\n\n"
        f"Conversation context:\n{context or 'N/A'}\n\n"
        "Use tools when entity resolution depends on live UE context or semantic knowledge.\n\n"
        f"Session ID: {session_id}\n"
        f"Snapshot ID: {snapshot_id}"
    )


def _build_record(pair: ArtifactPair) -> SupervisedSftRecord:
    target = OperationIntent.model_validate(pair.response.get("payload", {})).model_dump(mode="json")
    request = pair.request
    return SupervisedSftRecord(
        sample_id=f"iea:{pair.response.get('artifact_id')}",
        task="iea_sft",
        agent="intent_encoding",
        messages=[DatasetMessage(role="user", content=_format_request_message(request))],
        target=target,
        metadata={
            "matched_by": pair.matched_by,
            "request_artifact_id": request.get("artifact_id"),
            "response_artifact_id": pair.response.get("artifact_id"),
            "correlation_id": pair.response.get("correlation_id"),
        },
    )


def build_iea_sft_records(project_root: Path) -> Tuple[List[SupervisedSftRecord], List[Dict[str, Any]], BuildReport]:
    records, rejects, artifact_total = build_canonical_dataset(
        CanonicalBuilderConfig(
            project_root=project_root,
            request_relative="runtime/interfaces/coordinator__intent_encoding/requests",
            response_relative="runtime/interfaces/intent_encoding__coordinator/responses",
            build_record=_build_record,
        )
    )
    report = BuildReport(
        artifact_total=artifact_total,
        paired_total=len(records),
        iea_sft_samples=len(records),
        reject_total=len(rejects),
    )
    return records, rejects, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the IEA supervised SFT dataset.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=dataset_output_path("intent_encoding", "supervised", "success", "iea_sft_v1.jsonl"),
    )
    parser.add_argument(
        "--reject-output",
        type=Path,
        default=dataset_output_path("intent_encoding", "supervised", "failure", "iea_sft_rejects_v1.jsonl"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=dataset_dir("intent_encoding", "supervised") / "build_report_v1.json",
    )
    args = parser.parse_args()

    records, rejects, report = build_iea_sft_records(args.project_root)
    write_jsonl(args.output, records)
    write_jsonl(args.reject_output, rejects)
    merged_report = load_build_report(args.report_output)
    merged_report.artifact_total = report.artifact_total
    merged_report.paired_total = report.paired_total
    merged_report.iea_sft_samples = report.iea_sft_samples
    merged_report.reject_total = report.reject_total
    save_build_report(args.report_output, merged_report)
    print(f"Wrote {len(records)} IEA SFT rows to {args.output}")
    print(f"Rejected {len(rejects)} rows")


if __name__ == "__main__":
    main()

