from __future__ import annotations

from pathlib import Path

from .contracts import LaunchPlan


class EnvironmentLauncher:
    """Build launch plans matching experiments/scripts/launch_experiments.py."""

    def __init__(
        self,
        *,
        project_root: Path,
        workspace_root: Path,
        python_executable: Path,
        stack_python_executable: Path,
    ) -> None:
        self.project_root = Path(project_root)
        self.workspace_root = Path(workspace_root)
        self.python_executable = Path(python_executable)
        self.stack_python_executable = Path(stack_python_executable)
        self.ns3_integration_root = self.workspace_root / "ns3-free5gc-integration"

    def build_direct_launch_plan(
        self,
        *,
        scenario_path: Path,
        run_id: str,
        live_graph_snapshot_id: str,
        graph_db_url: str,
    ) -> LaunchPlan:
        run_dir = self.ns3_integration_root / "artifacts" / "runs" / run_id
        manifest_path = run_dir / "run-manifest.split.json"
        start_script = self.ns3_integration_root / "scripts" / "start_split_mode.py"
        argv: list[object] = [
            self.stack_python_executable,
            start_script,
            Path(scenario_path),
            "--run-id",
            run_id,
            "--live-graph-snapshot-id",
            live_graph_snapshot_id,
            "--wait-background",
        ]
        return LaunchPlan(
            cwd=self.ns3_integration_root,
            argv=argv,
            graph_db_url=graph_db_url,
            live_graph_snapshot_id=live_graph_snapshot_id,
            manifest_path=manifest_path,
            metadata={
                "launcher": "direct_split_mode",
                "scenario_path": str(Path(scenario_path)),
                "run_id": run_id,
            },
        )

    def build_registered_experiment_command(
        self,
        *,
        experiment_id: str,
        scenario_id: str,
        method_id: str = "Ours",
        start_index: int = 1,
    ) -> list[object]:
        return [
            self.python_executable,
            self.project_root / "experiments" / "scripts" / "launch_experiments.py",
            "--experiment",
            experiment_id,
            "--scenario",
            scenario_id,
            "--method",
            method_id,
            "--start-index",
            str(start_index),
        ]
