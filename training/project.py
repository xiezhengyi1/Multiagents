from __future__ import annotations

import argparse
from pathlib import Path

from training.artifacts import processed_dir, read_jsonl, write_jsonl
from training.projection import (
    COLLABORATION_AGENT_NAMES,
    WORKFLOW_AGENT_NAMES,
    build_chatml_record,
    build_finetuning_record,
    build_workflow_record,
    index_traces_by_session,
    load_traces,
)
from training.schemas import FineTuningTraceRecord


def project_agents(*, agents: list[str], output_dir: Path | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for agent_name, traces in load_traces(agents).items():
        output_path = (output_dir or processed_dir(agent_name)) / f"{agent_name}_trajectories_v1.jsonl"
        counts[agent_name] = write_jsonl(
            output_path,
            (build_finetuning_record(trace) for trace in traces),
        )
    return counts


def project_workflows(*, run_records_path: Path, output_path: Path) -> int:
    traces_by_session = index_traces_by_session(load_traces(WORKFLOW_AGENT_NAMES))
    records = (
        build_workflow_record(
            record,
            session_traces=traces_by_session.get(str(record.get("session_id") or "").strip(), {}),
        )
        for record in read_jsonl(run_records_path)
    )
    return write_jsonl(output_path, records)


def export_chatml(*, input_path: Path, output_path: Path) -> int:
    traces = (FineTuningTraceRecord.model_validate(row) for row in read_jsonl(input_path))
    return write_jsonl(output_path, (build_chatml_record(trace) for trace in traces))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project raw traces into training artifacts.")
    commands = parser.add_subparsers(dest="command", required=True)

    agents = commands.add_parser("agents", help="Project per-agent fine-tuning trajectories.")
    agents.add_argument("--agent", dest="agents", action="append")
    agents.add_argument("--output-dir", type=Path)

    workflows = commands.add_parser("workflows", help="Join workflow runs with agent traces.")
    workflows.add_argument(
        "--run-records",
        type=Path,
        default=processed_dir("workflow") / "trajectory_runs_v1.jsonl",
    )
    workflows.add_argument(
        "--output",
        type=Path,
        default=processed_dir("workflow") / "workflow_trajectories_v1.jsonl",
    )

    chatml = commands.add_parser("chatml", help="Export fine-tuning trajectories as ChatML.")
    chatml.add_argument("--agent", required=True)
    chatml.add_argument("--input", type=Path)
    chatml.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "agents":
        agents = list(dict.fromkeys(args.agents or COLLABORATION_AGENT_NAMES))
        counts = project_agents(agents=agents, output_dir=args.output_dir)
        for agent_name, count in counts.items():
            print(f"{agent_name}: {count} trajectories")
    elif args.command == "workflows":
        count = project_workflows(run_records_path=args.run_records, output_path=args.output)
        print(f"workflow trajectories: {count}")
    else:
        input_path = args.input or processed_dir(args.agent) / f"{args.agent}_trajectories_v1.jsonl"
        output_path = args.output or processed_dir(args.agent) / f"{args.agent}_chatml_v1.jsonl"
        count = export_chatml(input_path=input_path, output_path=output_path)
        print(f"{args.agent} chatml records: {count}")


if __name__ == "__main__":
    main()
