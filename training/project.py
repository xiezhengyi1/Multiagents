from __future__ import annotations

import argparse
from pathlib import Path

from training.artifacts import processed_dir, read_jsonl, write_jsonl
from training.projection import (
    COLLABORATION_AGENT_NAMES,
    WORKFLOW_AGENT_NAMES,
    build_qwen3_record,
    build_sft_example,
    build_workflow_record,
    index_traces_by_session,
    is_sft_candidate,
    load_traces,
)
from training.schemas import SftExampleV2


def project_agents(*, agents: list[str], output_dir: Path | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for agent_name, traces in load_traces(agents).items():
        candidates = [trace for trace in traces if is_sft_candidate(trace)]
        output_path = (output_dir or processed_dir(agent_name)) / f"{agent_name}_sft_v2.jsonl"
        counts[agent_name] = write_jsonl(
            output_path,
            (build_sft_example(trace) for trace in candidates),
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


def export_qwen3(*, input_path: Path, output_path: Path) -> int:
    examples = (SftExampleV2.model_validate(row) for row in read_jsonl(input_path))
    return write_jsonl(output_path, (build_qwen3_record(example) for example in examples))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project Trace V2 records into validated SFT artifacts.")
    commands = parser.add_subparsers(dest="command", required=True)

    agents = commands.add_parser("agents", help="Project validated model-independent SFT examples.")
    agents.add_argument("--agent", dest="agents", action="append")
    agents.add_argument("--output-dir", type=Path)

    workflows = commands.add_parser("workflows", help="Join workflow runs with agent traces.")
    workflows.add_argument(
        "--run-records",
        type=Path,
        default=processed_dir("workflow") / "trajectory_runs_v2.jsonl",
    )
    workflows.add_argument(
        "--output",
        type=Path,
        default=processed_dir("workflow") / "workflow_trajectories_v2.jsonl",
    )

    qwen3 = commands.add_parser("qwen3", help="Export validated SFT examples for Qwen3 tool calling.")
    qwen3.add_argument("--agent", required=True)
    qwen3.add_argument("--input", type=Path)
    qwen3.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "agents":
        agents = list(dict.fromkeys(args.agents or COLLABORATION_AGENT_NAMES))
        counts = project_agents(agents=agents, output_dir=args.output_dir)
        for agent_name, count in counts.items():
            print(f"{agent_name}: {count} validated SFT examples")
    elif args.command == "workflows":
        count = project_workflows(run_records_path=args.run_records, output_path=args.output)
        print(f"workflow trajectories: {count}")
    else:
        input_path = args.input or processed_dir(args.agent) / f"{args.agent}_sft_v2.jsonl"
        output_path = args.output or processed_dir(args.agent) / f"{args.agent}_qwen3_sft_v2.jsonl"
        count = export_qwen3(input_path=input_path, output_path=output_path)
        print(f"{args.agent} Qwen3 SFT records: {count}")


if __name__ == "__main__":
    main()
