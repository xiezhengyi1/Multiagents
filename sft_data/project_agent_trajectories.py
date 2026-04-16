from __future__ import annotations

import argparse
from pathlib import Path

from sft_data.common import ensure_agent_layout, processed_dir, repo_root
from sft_data.schemas import write_jsonl
from sft_data.trajectory_projection import (
    COLLABORATION_AGENT_NAMES,
    load_projected_traces_by_agent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project agent raw run-tree traces into tool-bearing trajectory records.",
    )
    parser.add_argument(
        "--agent",
        dest="agents",
        action="append",
        default=None,
        help="Agent name to project. Repeat to project multiple agents.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional custom output root. Defaults to each agent's processed directory.",
    )
    return parser.parse_args()


def project_agent_trajectories(*, agents: list[str], output_dir: Path | None) -> dict[str, int]:
    traces_by_agent = load_projected_traces_by_agent(repo_root(), agent_names=agents)
    counts: dict[str, int] = {}
    for agent_name, traces in traces_by_agent.items():
        ensure_agent_layout(agent_name)
        output_path = (
            output_dir / f"{agent_name}_trajectories_v1.jsonl"
            if output_dir is not None
            else processed_dir(agent_name) / f"{agent_name}_trajectories_v1.jsonl"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = write_jsonl(output_path, traces)
        counts[agent_name] = count
    return counts


def main() -> None:
    args = parse_args()
    raw_agents = args.agents if args.agents is not None else list(COLLABORATION_AGENT_NAMES)
    agents: list[str] = []
    for agent in raw_agents:
        normalized = str(agent).strip()
        if normalized and normalized not in agents:
            agents.append(normalized)
    if not agents:
        raise ValueError("At least one --agent value is required")
    counts = project_agent_trajectories(agents=agents, output_dir=args.output_dir)
    for agent_name, count in counts.items():
        print(f"{agent_name}: {count} trajectories")


if __name__ == "__main__":
    main()
