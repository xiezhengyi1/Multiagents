from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from experiments.scripts.common import PROJECT_ROOT, WORKSPACE_ROOT, build_project_python_env, resolve_python_executable
from experiments.scripts.simulator import start_simulator_for_scenario, stop_simulator


PYTHON_EXE = resolve_python_executable(PROJECT_ROOT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start simulator first, then start the autonomous guarded multi-agent system.",
    )
    parser.add_argument("--experiment", default="E1", help="Experiment id used to derive live snapshot id, e.g. E1.")
    parser.add_argument("--scenario", default="S2", help="Scenario id to start in the simulator, e.g. S2.")
    parser.add_argument("--initial-input", default="", help="Optional first user input injected when the watch loop starts.")
    parser.add_argument("--max-rounds", type=int, default=3, help="Max rounds inside MainControlOrchestrator.")
    parser.add_argument("--watch-interval", type=float, default=1.0, help="Seconds between watch-loop ticks.")
    parser.add_argument("--watch-iterations", type=int, default=0, help="0 means guard continuously.")
    parser.add_argument("--monitor-context-chars", type=int, default=4000, help="Context budget for monitor-triggered reentry.")
    parser.add_argument("--deepseek", action="store_true", help="Use deepseek-v4-flash for all control agents.")
    parser.add_argument("--keep-live-graph", action="store_true", help="Do not delete the previous live graph snapshot before startup.")
    return parser.parse_args()


def _run_project_command(command: list[str]) -> None:
    subprocess.run(command, cwd=WORKSPACE_ROOT, env=build_project_python_env(), check=True)


def main() -> None:
    args = _parse_args()
    simulator_state = None
    try:
        simulator_state = start_simulator_for_scenario(
            str(args.experiment or "").strip(),
            str(args.scenario or "").strip(),
            reset_live_graph=not bool(args.keep_live_graph),
        )
        snapshot_id = str(simulator_state["live_graph_snapshot_id"])
        _run_project_command(
            [
                str(PYTHON_EXE),
                "-m",
                "control_runtime.integrations.scenario.init_scenario",
                "--graph-snapshot-id",
                snapshot_id,
            ]
        )
        command = [
            str(PYTHON_EXE),
            "-m",
            "control_runtime.orchestrators.main_control_orchestrator",
            "--watch",
            "--snapshot-id",
            snapshot_id,
            "--scenario-id",
            str(args.experiment or "").strip(),
            "--scenario-tag",
            "guarded_system",
            "--scenario-tag",
            str(args.scenario or "").strip(),
            "--max-rounds",
            str(args.max_rounds),
            "--watch-interval",
            str(args.watch_interval),
            "--monitor-context-chars",
            str(args.monitor_context_chars),
        ]
        if int(args.watch_iterations or 0) > 0:
            command.extend(["--watch-iterations", str(args.watch_iterations)])
        if args.deepseek:
            command.append("--deepseek")
        initial_input = str(args.initial_input or "").strip()
        if initial_input:
            command.append(initial_input)
        print(
            json.dumps(
                {
                    "stage": "start_guarded_system",
                    "snapshot_id": snapshot_id,
                    "scenario": args.scenario,
                    "command": command,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        _run_project_command(command)
    finally:
        if simulator_state is not None:
            stop_simulator(simulator_state, reason="guarded system stopped")


if __name__ == "__main__":
    main()

