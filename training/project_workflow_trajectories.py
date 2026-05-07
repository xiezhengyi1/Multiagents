from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from training.common import ensure_agent_layout, processed_dir, repo_root
from training.schemas import WorkflowTrajectoryRecord, write_jsonl
from training.trajectory_projection import (
    WORKFLOW_AGENT_NAMES,
    build_workflow_trajectory_record,
    index_projected_traces_by_session,
    load_projected_traces_by_agent,
    read_jsonl_objects,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate workflow runs with session-linked agent trajectories and flattened tool traces.",
    )
    parser.add_argument(
        "--run-records",
        type=Path,
        default=processed_dir("workflow") / "trajectory_runs_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=processed_dir("workflow") / "workflow_trajectories_v1.jsonl",
    )
    return parser.parse_args()


def project_workflow_trajectories(*, run_records_path: Path, output_path: Path) -> int:
    ensure_agent_layout("workflow")
    run_records = read_jsonl_objects(run_records_path)
    traces_by_agent = load_projected_traces_by_agent(repo_root(), agent_names=WORKFLOW_AGENT_NAMES)
    traces_by_session = index_projected_traces_by_session(traces_by_agent)

    workflow_records: list[WorkflowTrajectoryRecord] = []
    for record in run_records:
        session_id = str(record.get("session_id") or "").strip()
        if not session_id:
            raise ValueError(f"Workflow record #{record.get('record_index')} is missing session_id")
        workflow_records.append(
            build_workflow_trajectory_record(
                record,
                session_traces=traces_by_session.get(session_id, {}),
                agent_order=WORKFLOW_AGENT_NAMES,
            )
        )
    return write_jsonl(output_path, workflow_records)


def main() -> None:
    args = parse_args()
    count = project_workflow_trajectories(
        run_records_path=args.run_records,
        output_path=args.output,
    )
    print(f"workflow trajectories: {count}")


if __name__ == "__main__":
    main()
