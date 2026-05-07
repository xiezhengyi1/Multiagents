from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from training.common import ensure_agent_layout, load_trace_records, processed_dir
from training.schemas import ChatmlSftRecord, ProjectedTraceRecord, write_jsonl
from training.trajectory_projection import build_chatml_record_from_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert projected agent trajectories into ChatML JSONL records.",
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent name whose projected trajectories will be converted.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional projected trajectory JSONL input. Defaults to training/<agent>/processed/<agent>_trajectories_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional ChatML JSONL output. Defaults to training/<agent>/processed/<agent>_chatml_v1.jsonl",
    )
    return parser.parse_args()


def export_agent_trajectories_to_chatml(*, agent_name: str, input_path: Path, output_path: Path) -> int:
    ensure_agent_layout(agent_name)
    traces = load_trace_records(input_path, ProjectedTraceRecord)
    chatml_records: list[ChatmlSftRecord] = [
        build_chatml_record_from_trace(trace)
        for trace in traces
    ]
    return write_jsonl(output_path, chatml_records)


def main() -> None:
    args = parse_args()
    agent_name = str(args.agent or "").strip()
    if not agent_name:
        raise ValueError("--agent is required")

    input_path = args.input or (processed_dir(agent_name) / f"{agent_name}_trajectories_v1.jsonl")
    output_path = args.output or (processed_dir(agent_name) / f"{agent_name}_chatml_v1.jsonl")

    count = export_agent_trajectories_to_chatml(
        agent_name=agent_name,
        input_path=input_path,
        output_path=output_path,
    )
    print(f"{agent_name} chatml records: {count}")


if __name__ == "__main__":
    main()
