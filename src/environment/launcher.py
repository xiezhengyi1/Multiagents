from __future__ import annotations

import csv
import http.client
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

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
        process_factory: Callable[..., Any] = subprocess.Popen,
        command_runner: Callable[..., Any] = subprocess.run,
        graph_snapshot_probe: Callable[[LaunchPlan], dict[str, Any]] | None = None,
        gateway_probe: Callable[[LaunchPlan], dict[str, Any]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.project_root = Path(project_root)
        self.workspace_root = Path(workspace_root)
        self.python_executable = Path(python_executable)
        self.stack_python_executable = Path(stack_python_executable)
        self.ns3_integration_root = self.workspace_root / "ns3-free5gc-integration"
        self.process_factory = process_factory
        self.command_runner = command_runner
        self.graph_snapshot_probe = graph_snapshot_probe or self._probe_graph_snapshot
        self.gateway_probe = gateway_probe or self._probe_gateway
        self.sleep = sleep
        self.monotonic = monotonic

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

    def validate_sla_initialization(self, manifest: dict[str, Any]) -> dict[str, Any]:
        profile_path = Path(str(manifest.get("flow_profile_file") or ""))
        errors: list[str] = []
        if not profile_path.is_file():
            return {"ok": False, "flow_count": 0, "errors": [f"missing SLA flow profile: {profile_path}"]}

        required_columns = {
            "flow_id",
            "bandwidth_dl_mbps",
            "bandwidth_ul_mbps",
            "guaranteed_bandwidth_dl_mbps",
            "guaranteed_bandwidth_ul_mbps",
            "priority",
            "allocated_bandwidth_dl_mbps",
            "allocated_bandwidth_ul_mbps",
        }
        with profile_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing_columns = sorted(required_columns - set(reader.fieldnames or ()))
            if missing_columns:
                return {
                    "ok": False,
                    "flow_count": 0,
                    "errors": ["missing SLA profile columns: " + ", ".join(missing_columns)],
                }
            rows = list(reader)

        if not rows:
            return {"ok": False, "flow_count": 0, "errors": ["SLA flow profile has no initialized flows"]}

        for row in rows:
            flow_id = str(row.get("flow_id") or "<unknown>")
            try:
                target_dl = float(row["bandwidth_dl_mbps"])
                target_ul = float(row["bandwidth_ul_mbps"])
                guaranteed_dl = float(row["guaranteed_bandwidth_dl_mbps"])
                guaranteed_ul = float(row["guaranteed_bandwidth_ul_mbps"])
                allocated_dl = float(row["allocated_bandwidth_dl_mbps"])
                allocated_ul = float(row["allocated_bandwidth_ul_mbps"])
                int(row["priority"])
            except (TypeError, ValueError) as exc:
                errors.append(f"flow {flow_id} has invalid SLA initialization values: {exc}")
                continue
            if guaranteed_dl < 0 or guaranteed_ul < 0:
                errors.append(f"flow {flow_id} has negative guaranteed bandwidth")
            if target_dl < guaranteed_dl or target_ul < guaranteed_ul:
                errors.append(f"flow {flow_id} target bandwidth is below guaranteed bandwidth")
            if allocated_dl < guaranteed_dl:
                errors.append(
                    f"flow {flow_id} allocated DL bandwidth {allocated_dl} "
                    f"is below guaranteed DL bandwidth {guaranteed_dl}"
                )
            if allocated_ul < guaranteed_ul:
                errors.append(
                    f"flow {flow_id} allocated UL bandwidth {allocated_ul} "
                    f"is below guaranteed UL bandwidth {guaranteed_ul}"
                )
        return {"ok": not errors, "flow_count": len(rows), "errors": errors}

    def validate_simulator_startup(
        self,
        plan: LaunchPlan,
        *,
        poll_interval_seconds: float = 1.0,
    ) -> dict[str, Any]:
        plan.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        plan.manifest_path.unlink(missing_ok=True)
        launch_log = plan.manifest_path.parent / "environment-agent-validation.log"
        process: Any = None
        last_readiness: dict[str, Any] = {}
        with launch_log.open("a", encoding="utf-8") as handle:
            try:
                process = self.process_factory(
                    [str(item) for item in plan.argv],
                    cwd=str(plan.cwd),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                deadline = self.monotonic() + max(0.0, float(plan.timeout_seconds))
                while self.monotonic() <= deadline:
                    return_code = process.poll()
                    if return_code is not None:
                        return {
                            "status": "failed",
                            "simulator_started": True,
                            "error": f"simulator launcher exited before readiness with code {return_code}",
                            "readiness": last_readiness,
                            "log_tail": self._read_log_tail(launch_log),
                        }
                    last_readiness = self._collect_readiness(plan)
                    if last_readiness["ok"]:
                        return {
                            "status": "ok",
                            "simulator_started": True,
                            "manifest_path": str(plan.manifest_path),
                            "graph_snapshot": last_readiness["graph_snapshot"],
                            "gateway_health": last_readiness["gateway_health"],
                            "sla_initialization": last_readiness["sla_initialization"],
                        }
                    self.sleep(max(0.0, float(poll_interval_seconds)))
                return {
                    "status": "failed",
                    "simulator_started": True,
                    "error": "simulator readiness timed out",
                    "readiness": last_readiness,
                    "log_tail": self._read_log_tail(launch_log),
                }
            except Exception as exc:
                return {
                    "status": "failed",
                    "simulator_started": process is not None,
                    "error": f"simulator validation failed: {exc}",
                    "readiness": last_readiness,
                    "log_tail": self._read_log_tail(launch_log),
                }
            finally:
                if process is not None:
                    self._cleanup_validation_process(process, plan.manifest_path)

    def _collect_readiness(self, plan: LaunchPlan) -> dict[str, Any]:
        if not plan.manifest_path.is_file():
            return {
                "ok": False,
                "manifest_exists": False,
                "graph_snapshot": {"ok": False, "exists": False},
                "gateway_health": {"ok": False, "healthy": False},
                "sla_initialization": {"ok": False, "flow_count": 0, "errors": ["manifest not ready"]},
            }
        manifest = json.loads(plan.manifest_path.read_text(encoding="utf-8"))
        graph_snapshot = self.graph_snapshot_probe(plan)
        gateway_health = self.gateway_probe(plan)
        sla_initialization = self.validate_sla_initialization(manifest)
        return {
            "ok": bool(graph_snapshot.get("ok") and gateway_health.get("ok") and sla_initialization.get("ok")),
            "manifest_exists": True,
            "graph_snapshot": graph_snapshot,
            "gateway_health": gateway_health,
            "sla_initialization": sla_initialization,
        }

    def _probe_graph_snapshot(self, plan: LaunchPlan) -> dict[str, Any]:
        if not plan.graph_db_url:
            return {"ok": False, "exists": False, "error": "graph_db_url is required"}
        writer_cli = self.ns3_integration_root / "bridge" / "writer" / "cli.py"
        completed = self.command_runner(
            [
                str(self.stack_python_executable),
                str(writer_cli),
                "graph-snapshot-exists",
                "--graph-db-url",
                plan.graph_db_url,
                "--snapshot-id",
                plan.live_graph_snapshot_id,
                "--ensure-graph-schema",
            ],
            cwd=str(self.ns3_integration_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return {"ok": False, "exists": False, "error": completed.stderr.strip()}
        payload = json.loads(completed.stdout.strip() or "{}")
        exists = bool(payload.get("exists"))
        return {"ok": exists, "exists": exists, "snapshot_id": plan.live_graph_snapshot_id}

    @staticmethod
    def _probe_gateway(plan: LaunchPlan) -> dict[str, Any]:
        connection = http.client.HTTPConnection(plan.healthcheck_host, plan.healthcheck_port, timeout=2.0)
        try:
            connection.request("GET", plan.healthcheck_path)
            response = connection.getresponse()
            body = response.read().decode("utf-8", errors="replace")
            payload = json.loads(body or "{}")
        except Exception as exc:
            return {"ok": False, "healthy": False, "error": str(exc)}
        finally:
            connection.close()
        required = ("healthy", "flow_profile_exists", "latest_snapshot_exists", "upstream_ok")
        ok = response.status == 200 and all(payload.get(key) is True for key in required)
        return {"ok": ok, "http_status": response.status, **payload}

    def _cleanup_validation_process(self, process: Any, manifest_path: Path) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._run_compose_down_from_manifest(manifest_path)

    def _run_compose_down_from_manifest(self, manifest_path: Path) -> None:
        if not manifest_path.is_file():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        command = next((item for item in manifest.get("commands", []) if item.get("name") == "compose-down"), None)
        if not command:
            return
        self.command_runner(
            [str(item) for item in command["argv"]],
            cwd=str(command["cwd"]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    @staticmethod
    def _read_log_tail(path: Path, line_count: int = 30) -> str:
        if not path.is_file():
            return ""
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:])

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
