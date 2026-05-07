from __future__ import annotations

import argparse
from datetime import datetime
import http.client
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from experiments.scripts.common import (
    CONFIG_ROOT,
    EXPERIMENT_ROOT,
    LEDGER_ROOT,
    PROJECT_ROOT,
    WORKSPACE_ROOT,
    load_json,
    load_yaml_mapping,
    resolve_python_executable,
)
from experiments.paths import resolve_scenario_source_path, scoped_catalog_input_path


MATRIX_PATH = CONFIG_ROOT / "experiment_matrix.json"
METHODS_PATH = CONFIG_ROOT / "methods.json"
PYTHON_EXE = resolve_python_executable(PROJECT_ROOT)
BUILD_INPUTS_SCRIPT = EXPERIMENT_ROOT / "scripts" / "build_user_inputs.py"
RUN_METHOD_SCRIPT = EXPERIMENT_ROOT / "scripts" / "run_method.py"
AGGREGATE_SCRIPT = EXPERIMENT_ROOT / "scripts" / "aggregate_results.py"
FAILED_RUNS_PATH = LEDGER_ROOT / "failed_experiments.jsonl"
SIMULATOR_LOG_DIR = LEDGER_ROOT / "simulator"
NS3_INTEGRATION_ROOT = WORKSPACE_ROOT / "ns3-free5gc-integration"
START_SIMULATOR_SCRIPT = NS3_INTEGRATION_ROOT / "scripts" / "start_split_mode.py"
GRAPH_WRITER_CLI = NS3_INTEGRATION_ROOT / "bridge" / "writer" / "cli.py"
STACK_PYTHON_EXE = resolve_python_executable(NS3_INTEGRATION_ROOT)
SIMULATOR_BOOTSTRAP_TIMEOUT_SEC = 180.0
SIMULATOR_GATEWAY_HOST = "127.0.0.1"
SIMULATOR_GATEWAY_PORT = 18080
SIMULATOR_GATEWAY_PROBE_PATH = "/policy-executions/launch-healthcheck"
SIMULATOR_GATEWAY_TIMEOUT_SEC = 2.0
SIMULATOR_RESTART_LIMIT = 2

def _load_matrix() -> List[Dict[str, Any]]:
    payload = load_json(MATRIX_PATH)
    experiments = payload.get("experiments", [])
    if not isinstance(experiments, list):
        raise TypeError("experiment_matrix.json experiments must be a list")
    return experiments


def _load_methods() -> Dict[str, Dict[str, Any]]:
    payload = load_json(METHODS_PATH)
    methods = payload.get("methods", [])
    if not isinstance(methods, list):
        raise TypeError("methods.json methods must be a list")
    return {
        str(item.get("id") or "").strip(): item
        for item in methods
        if str(item.get("id") or "").strip()
    }


def _run_command(
    command: List[str],
    *,
    label: str,
    cwd: Path = WORKSPACE_ROOT,
    emit_output: bool = True,
    stream_output: bool = False,
) -> str:
    if emit_output:
        print(f"[launch] {label}", flush=True)
    if stream_output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        output_lines: List[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            if emit_output:
                print(line.rstrip("\n"), flush=True)
        return_code = process.wait()
        output_text = "".join(output_lines).strip()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command, output=output_text)
        return output_text
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    stdout = completed.stdout.strip()
    if emit_output and stdout:
        print(stdout, flush=True)
    stderr = completed.stderr.strip()
    if emit_output and stderr:
        print(stderr, file=sys.stderr, flush=True)
    return stdout


def _append_failed_run(record: Dict[str, Any]) -> None:
    FAILED_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FAILED_RUNS_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _resolve_scenario_path(scenario_id: str) -> Path:
    scenario_path = resolve_scenario_source_path(scenario_id)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario source not found for {scenario_id}: {scenario_path}")
    return scenario_path


def _resolve_simulator_scenario_path(scenario_id: str) -> Path:
    base_scenario_path = _resolve_scenario_path(scenario_id)
    split_mode_path = NS3_INTEGRATION_ROOT / "scenarios" / "split_mode" / base_scenario_path.name
    if split_mode_path.exists():
        return split_mode_path
    return base_scenario_path


def _resolve_graph_db_url(scenario_path: Path) -> str:
    payload = load_yaml_mapping(scenario_path)
    writer_payload = payload.get("writer") if isinstance(payload.get("writer"), dict) else {}
    topology_payload = payload.get("topology") if isinstance(payload.get("topology"), dict) else {}
    graph_db_url = str(
        writer_payload.get("graph_db_url")
        or topology_payload.get("graph_db_url")
        or ""
    ).strip()
    if not graph_db_url:
        base_scenario_value = payload.get("base_scenario")
        if isinstance(base_scenario_value, str) and base_scenario_value.strip():
            base_scenario_path = (scenario_path.parent / base_scenario_value).resolve()
            return _resolve_graph_db_url(base_scenario_path)
        raise ValueError(f"Scenario YAML does not define writer.graph_db_url or topology.graph_db_url: {scenario_path}")
    return graph_db_url


def _resolve_simulator_manifest_path(run_dir: Path) -> Path:
    split_manifest_path = run_dir / "run-manifest.split.json"
    if split_manifest_path.exists() or START_SIMULATOR_SCRIPT.name == "start_split_mode.py":
        return split_manifest_path
    return run_dir / "run-manifest.json"


def _build_experiment_live_graph_snapshot_id(experiment_id: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in str(experiment_id or "").strip())
    normalized = normalized.strip("-") or "experiment"
    return f"live-{normalized}"


def _run_graph_admin_command(arguments: List[str], *, label: str, emit_output: bool = True) -> Dict[str, Any]:
    stdout = _run_command(
        [str(STACK_PYTHON_EXE), str(GRAPH_WRITER_CLI), *arguments],
        label=label,
        cwd=NS3_INTEGRATION_ROOT,
        emit_output=emit_output,
    )
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse graph admin output for {label}: {stdout}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Graph admin output must be a JSON object for {label}: {payload}")
    return payload


def _graph_snapshot_exists(*, graph_db_url: str, snapshot_id: str) -> bool:
    payload = _run_graph_admin_command(
        [
            "graph-snapshot-exists",
            "--graph-db-url",
            graph_db_url,
            "--snapshot-id",
            snapshot_id,
            "--ensure-graph-schema",
        ],
        label=f"check graph snapshot {snapshot_id}",
        emit_output=False,
    )
    return bool(payload.get("exists"))


def _delete_graph_snapshot(*, graph_db_url: str, snapshot_id: str) -> Dict[str, Any]:
    return _run_graph_admin_command(
        [
            "delete-graph-snapshot",
            "--graph-db-url",
            graph_db_url,
            "--snapshot-id",
            snapshot_id,
            "--ensure-graph-schema",
        ],
        label=f"delete graph snapshot {snapshot_id}",
    )


def _prune_graph_snapshots(*, graph_db_url: str, keep_snapshot_id: str = "", keep_latest: bool = False) -> Dict[str, Any]:
    arguments = [
        "prune-graph-snapshots",
        "--graph-db-url",
        graph_db_url,
        "--ensure-graph-schema",
    ]
    if keep_snapshot_id:
        arguments.extend(["--keep-snapshot-id", keep_snapshot_id])
    if keep_latest:
        arguments.append("--keep-latest")
    return _run_graph_admin_command(
        arguments,
        label=(f"prune graph snapshots keep {keep_snapshot_id}" if keep_snapshot_id else "prune graph snapshots keep latest"),
    )


def _read_log_tail(path: Path, *, max_lines: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])


def _mirror_log_increment(*, source_path: Path, destination_path: Path, cursor: int = 0) -> int:
    if not source_path.exists():
        return cursor
    try:
        with source_path.open("r", encoding="utf-8", errors="ignore") as source_handle:
            source_handle.seek(cursor)
            chunk = source_handle.read()
            next_cursor = source_handle.tell()
    except OSError:
        return cursor
    if chunk:
        with destination_path.open("a", encoding="utf-8") as destination_handle:
            destination_handle.write(chunk)
    return next_cursor


def _run_compose_down_from_manifest(manifest_path: Path) -> None:
    if not manifest_path.exists():
        return
    payload = load_json(manifest_path)
    commands = payload.get("commands", [])
    compose_down = next((item for item in commands if str(item.get("name") or "") == "compose-down"), None)
    if not isinstance(compose_down, dict):
        return
    argv_items = [str(item) for item in compose_down.get("argv", [])]
    if not argv_items:
        return
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in (compose_down.get("env") or {}).items()})
    subprocess.run(
        argv_items,
        cwd=str(compose_down.get("cwd") or NS3_INTEGRATION_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _probe_simulator_gateway(*, timeout_sec: float = SIMULATOR_GATEWAY_TIMEOUT_SEC) -> tuple[bool, str]:
    connection = http.client.HTTPConnection(
        SIMULATOR_GATEWAY_HOST,
        SIMULATOR_GATEWAY_PORT,
        timeout=timeout_sec,
    )
    try:
        connection.request("GET", SIMULATOR_GATEWAY_PROBE_PATH)
        response = connection.getresponse()
        status_code = int(response.status)
        reason = str(response.reason or "").strip()
        response.read()
    except OSError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)
    finally:
        try:
            connection.close()
        except Exception:
            pass

    detail = f"http {status_code}{(' ' + reason) if reason else ''}"
    return (200 <= status_code < 500), detail


def _wait_for_simulator_bootstrap(
    *,
    process: subprocess.Popen[str],
    manifest_path: Path,
    log_path: Path,
    graph_db_url: str,
    live_graph_snapshot_id: str,
) -> None:
    deadline = time.monotonic() + SIMULATOR_BOOTSTRAP_TIMEOUT_SEC
    last_gateway_detail = "not checked"
    policy_acceptor_log_path = manifest_path.parent / "logs" / "policy-acceptor.log"
    policy_acceptor_cursor = 0
    while time.monotonic() < deadline:
        policy_acceptor_cursor = _mirror_log_increment(
            source_path=policy_acceptor_log_path,
            destination_path=log_path,
            cursor=policy_acceptor_cursor,
        )
        if process.poll() is not None:
            break
        graph_ready = manifest_path.exists() and _graph_snapshot_exists(
            graph_db_url=graph_db_url,
            snapshot_id=live_graph_snapshot_id,
        )
        gateway_ready, last_gateway_detail = _probe_simulator_gateway()
        if graph_ready and gateway_ready:
            return
        time.sleep(1.0)

    policy_acceptor_cursor = _mirror_log_increment(
        source_path=policy_acceptor_log_path,
        destination_path=log_path,
        cursor=policy_acceptor_cursor,
    )
    rc = process.poll()
    log_tail = _read_log_tail(log_path, max_lines=40)
    raise RuntimeError(
        "Simulator bootstrap did not become ready "
        f"(snapshot_id={live_graph_snapshot_id}, rc={rc}, gateway={last_gateway_detail}, manifest={manifest_path}, log={log_path})\n{log_tail}"
    )


def _start_simulator_for_scenario(
    experiment_id: str,
    scenario_id: str,
    *,
    reset_live_graph: bool = True,
) -> Dict[str, Any]:
    if not START_SIMULATOR_SCRIPT.exists():
        raise FileNotFoundError(f"Missing simulator launcher: {START_SIMULATOR_SCRIPT}")
    if not GRAPH_WRITER_CLI.exists():
        raise FileNotFoundError(f"Missing graph writer CLI: {GRAPH_WRITER_CLI}")
    scenario_path = _resolve_simulator_scenario_path(scenario_id)
    graph_db_url = _resolve_graph_db_url(scenario_path)
    live_graph_snapshot_id = _build_experiment_live_graph_snapshot_id(experiment_id)
    if reset_live_graph:
        _prune_graph_snapshots(graph_db_url=graph_db_url, keep_latest=True)
        _delete_graph_snapshot(graph_db_url=graph_db_url, snapshot_id=live_graph_snapshot_id)

    run_id = f"batch-{experiment_id.lower()}-{scenario_id.lower()}-{_timestamp()}"
    run_dir = NS3_INTEGRATION_ROOT / "artifacts" / "runs" / run_id
    manifest_path = _resolve_simulator_manifest_path(run_dir)
    SIMULATOR_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SIMULATOR_LOG_DIR / f"{run_id}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        str(STACK_PYTHON_EXE),
        str(START_SIMULATOR_SCRIPT),
        str(scenario_path),
        "--run-id",
        run_id,
        "--live-graph-snapshot-id",
        live_graph_snapshot_id,
        "--wait-background",
    ]
    print(
        f"[launch] start simulator for {experiment_id}/{scenario_id}: {scenario_path} "
        f"(live_graph_snapshot_id={live_graph_snapshot_id})",
        flush=True,
    )
    process = subprocess.Popen(
        command,
        cwd=NS3_INTEGRATION_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_simulator_bootstrap(
            process=process,
            manifest_path=manifest_path,
            log_path=log_path,
            graph_db_url=graph_db_url,
            live_graph_snapshot_id=live_graph_snapshot_id,
        )
        _prune_graph_snapshots(graph_db_url=graph_db_url, keep_snapshot_id=live_graph_snapshot_id)
    except Exception:
        try:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=20)
        except Exception:
            if process.poll() is None:
                process.kill()
        finally:
            log_handle.close()
            _run_compose_down_from_manifest(manifest_path)
        raise
    return {
        "process": process,
        "log_handle": log_handle,
        "log_path": log_path,
        "manifest_path": manifest_path,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "restart_count": 0,
        "scenario_id": scenario_id,
        "scenario_path": scenario_path,
        "graph_db_url": graph_db_url,
        "live_graph_snapshot_id": live_graph_snapshot_id,
    }


def _restart_simulator_for_current_scenario(state: Dict[str, Any], *, reason: str) -> Dict[str, Any]:
    next_restart_count = int(state.get("restart_count", 0) or 0) + 1
    print(
        f"[launch] restart simulator for {state['experiment_id']}/{state['scenario_id']} "
        f"(attempt={next_restart_count}, reason={reason})",
        flush=True,
    )
    _stop_simulator(state, reason=f"restart requested: {reason}")
    restarted_state = _start_simulator_for_scenario(
        state["experiment_id"],
        state["scenario_id"],
        reset_live_graph=False,
    )
    restarted_state["restart_count"] = next_restart_count
    return restarted_state


def _ensure_simulator_running(state: Dict[str, Any]) -> Dict[str, Any]:
    process = state["process"]
    rc = process.poll()
    if rc is None:
        gateway_ready, gateway_detail = _probe_simulator_gateway()
        if gateway_ready:
            return state
        failure_detail = (
            f"policy gateway unhealthy for {state['scenario_id']} "
            f"(run_id={state['run_id']}, live_graph_snapshot_id={state['live_graph_snapshot_id']}, gateway={gateway_detail})"
        )
    else:
        failure_detail = (
            f"Simulator exited early for {state['scenario_id']} "
            f"(run_id={state['run_id']}, live_graph_snapshot_id={state['live_graph_snapshot_id']}, rc={rc})"
        )

    restart_count = int(state.get("restart_count", 0) or 0)
    if restart_count < SIMULATOR_RESTART_LIMIT:
        return _restart_simulator_for_current_scenario(state, reason=failure_detail)

    log_tail = _read_log_tail(state["log_path"], max_lines=40)
    raise RuntimeError(f"{failure_detail}, log={state['log_path']}\n{log_tail}")


def _stop_simulator(state: Dict[str, Any] | None, *, reason: str) -> None:
    if not state:
        return
    process = state["process"]
    log_handle = state["log_handle"]
    manifest_path = state["manifest_path"]
    print(f"[launch] stop simulator for {state['scenario_id']} ({reason})", flush=True)
    graceful = False
    try:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=30)
        graceful = True
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
                graceful = True
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    finally:
        log_handle.close()
        if not graceful:
            _run_compose_down_from_manifest(manifest_path)


def _parse_run_method_output(stdout: str) -> Dict[str, Any]:
    text = str(stdout or "").strip()
    if not text:
        raise RuntimeError("run_method.py returned empty stdout")
    lines = [line for line in text.splitlines() if line.strip()]
    for start_index in range(len(lines)):
        candidate = "\n".join(lines[start_index:])
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("result_path") and payload.get("summary_path"):
            return payload
    raise RuntimeError("Failed to parse run_method.py JSON output")


def _iter_selected_experiments(
    experiments: List[Dict[str, Any]],
    *,
    experiment_filter: str,
) -> List[Dict[str, Any]]:
    if not experiment_filter:
        return experiments
    selected = [item for item in experiments if str(item.get("id") or "").strip() == experiment_filter]
    if not selected:
        raise ValueError(f"Unknown experiment id: {experiment_filter}")
    return selected


def _filter_values(values: List[str], allowed_filter: str) -> List[str]:
    normalized = [str(item).strip() for item in values if str(item).strip()]
    if not allowed_filter:
        return normalized
    selected = [item for item in normalized if item == allowed_filter]
    if not selected:
        raise ValueError(f"Requested filter value '{allowed_filter}' not found in {normalized}")
    return selected


def _generated_inputs_path(*, experiment_id: str, scenario_id: str) -> Path:
    return scoped_catalog_input_path(experiment_id=experiment_id, scenario_id=scenario_id)


def _load_generated_task_count(*, experiment_id: str, scenario_id: str, start_index: int = 1) -> int:
    path = _generated_inputs_path(experiment_id=experiment_id, scenario_id=scenario_id)
    if not path.exists():
        raise FileNotFoundError(f"Generated input file not found: {path}")
    payload = load_json(path)
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise TypeError(f"Generated input records must be a list: {path}")
    if start_index <= 0:
        raise ValueError("--start-index must be positive")
    remaining_count = len(records) - start_index + 1
    if remaining_count <= 0:
        raise RuntimeError(f"No generated input records found in {path} from start_index={start_index}")
    return remaining_count


def _init_experiment_summary(experiment_id: str, planned_scenarios: List[str], planned_methods: List[str]) -> Dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "planned_scenario_count": len(planned_scenarios),
        "planned_method_count": len(planned_methods),
        "planned_run_count": len(planned_scenarios) * len(planned_methods),
        "built_scenarios": {},
        "successful_runs": [],
        "failed_runs": [],
    }


def _print_experiment_progress(
    *,
    experiment_id: str,
    experiment_index: int,
    experiment_total: int,
    scenario_id: str,
    scenario_index: int,
    scenario_total: int,
    method_id: str = "",
    method_index: int = 0,
    method_total: int = 0,
    stage: str,
    task_count: int | None = None,
) -> None:
    parts = [
        f"experiment={experiment_id} ({experiment_index}/{experiment_total})",
        f"scenario={scenario_id} ({scenario_index}/{scenario_total})",
        f"stage={stage}",
    ]
    if method_id:
        parts.insert(2, f"agent={method_id} ({method_index}/{method_total})")
    if task_count is not None:
        parts.append(f"tasks={task_count}")
    print(f"[launch][progress] {' | '.join(parts)}", flush=True)


def _build_coverage_summary(experiment_summaries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    coverage: List[Dict[str, Any]] = []
    for experiment_id, summary in experiment_summaries.items():
        built_scenarios = summary["built_scenarios"]
        successful_runs = summary["successful_runs"]
        failed_runs = summary["failed_runs"]
        successful_scenarios = sorted({item["scenario_id"] for item in successful_runs})
        successful_methods = sorted({item["method_id"] for item in successful_runs})
        successful_task_invocations = sum(
            int(built_scenarios[item["scenario_id"]]["task_count"])
            for item in successful_runs
            if item["scenario_id"] in built_scenarios
        )
        coverage.append(
            {
                "experiment_id": experiment_id,
                "planned_scenario_count": summary["planned_scenario_count"],
                "planned_method_count": summary["planned_method_count"],
                "planned_run_count": summary["planned_run_count"],
                "built_scenario_count": len(built_scenarios),
                "built_scenarios": [
                    {
                        "scenario_id": scenario_id,
                        "task_count": scenario_meta["task_count"],
                        "generated_input_path": scenario_meta["generated_input_path"],
                    }
                    for scenario_id, scenario_meta in sorted(built_scenarios.items())
                ],
                "successful_run_count": len(successful_runs),
                "failed_run_count": len(failed_runs),
                "successful_scenario_count": len(successful_scenarios),
                "successful_method_count": len(successful_methods),
                "successful_scenarios": successful_scenarios,
                "successful_methods": successful_methods,
                "successful_task_invocations": successful_task_invocations,
            }
        )
    return coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch experiment runs strictly according to experiment_matrix.json.",
    )
    parser.add_argument("--experiment", default="", help="Only launch one experiment id, e.g. E1")
    parser.add_argument("--scenario", default="", help="Only launch one scenario id, e.g. S2")
    parser.add_argument("--method", default="", help="Only launch one method id, e.g. Ours")
    parser.add_argument("--start-index", type=int, default=1, help="1-based user_input record index to start from.")
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Run build_user_inputs.py and run_method.py only, without ledger aggregation.",
    )
    parser.add_argument(
        "--deepseek",
        action="store_true",
        dest="use_deepseek",
        help="Use deepseek-v4-flash for all agents (Ours variants only).",
    )
    return parser.parse_args()


def main() -> None:
    if not PYTHON_EXE.exists():
        raise FileNotFoundError(f"Missing Python executable: {PYTHON_EXE}")

    args = parse_args()
    start_index = int(args.start_index)
    if start_index <= 0:
        raise ValueError("--start-index must be positive")
    experiments = _iter_selected_experiments(
        _load_matrix(),
        experiment_filter=str(args.experiment or "").strip(),
    )
    methods_registry = _load_methods()

    launched_runs: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, Any]] = []
    experiment_summaries: Dict[str, Dict[str, Any]] = {}
    experiment_total = len(experiments)
    for experiment_index, experiment in enumerate(experiments, start=1):
        experiment_id = str(experiment.get("id") or "").strip()
        scenarios = _filter_values(
            list(experiment.get("scenarios") or []),
            str(args.scenario or "").strip(),
        )
        methods = _filter_values(
            list(experiment.get("methods") or []),
            str(args.method or "").strip(),
        )
        if not scenarios:
            raise RuntimeError(f"Experiment {experiment_id} has no scenarios")
        if not methods:
            raise RuntimeError(f"Experiment {experiment_id} has no methods")
        experiment_summaries[experiment_id] = _init_experiment_summary(experiment_id, scenarios, methods)
        scenario_total = len(scenarios)
        method_total = len(methods)

        for scenario_index, scenario_id in enumerate(scenarios, start=1):
            _print_experiment_progress(
                experiment_id=experiment_id,
                experiment_index=experiment_index,
                experiment_total=experiment_total,
                scenario_id=scenario_id,
                scenario_index=scenario_index,
                scenario_total=scenario_total,
                stage="build-inputs",
            )
            try:
                _run_command(
                    [
                        str(PYTHON_EXE),
                        str(BUILD_INPUTS_SCRIPT),
                        "--experiment",
                        experiment_id,
                        "--scenario",
                        scenario_id,
                    ],
                    label=f"build inputs for {experiment_id}/{scenario_id}",
                )
                task_count = _load_generated_task_count(
                    experiment_id=experiment_id,
                    scenario_id=scenario_id,
                    start_index=start_index,
                )
                experiment_summaries[experiment_id]["built_scenarios"][scenario_id] = {
                    "task_count": task_count,
                    "generated_input_path": str(
                        _generated_inputs_path(experiment_id=experiment_id, scenario_id=scenario_id)
                    ),
                }
                _print_experiment_progress(
                    experiment_id=experiment_id,
                    experiment_index=experiment_index,
                    experiment_total=experiment_total,
                    scenario_id=scenario_id,
                    scenario_index=scenario_index,
                    scenario_total=scenario_total,
                    stage="start-simulator",
                    task_count=task_count,
                )
            except Exception as exc:
                failure = {
                    "stage": "build_inputs",
                    "experiment_id": experiment_id,
                    "scenario_id": scenario_id,
                    "method_id": "",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failed_runs.append(failure)
                experiment_summaries[experiment_id]["failed_runs"].append(failure)
                _append_failed_run(failure)
                print(
                    f"[launch][skip] build inputs failed for {experiment_id}/{scenario_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            simulator_state: Dict[str, Any] | None = None
            try:
                simulator_state = _start_simulator_for_scenario(experiment_id, scenario_id)
            except Exception as exc:
                failure = {
                    "stage": "start_simulator",
                    "experiment_id": experiment_id,
                    "scenario_id": scenario_id,
                    "method_id": "",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failed_runs.append(failure)
                experiment_summaries[experiment_id]["failed_runs"].append(failure)
                _append_failed_run(failure)
                print(
                    f"[launch][skip] simulator startup failed for {experiment_id}/{scenario_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            try:
                for method_index, method_id in enumerate(methods, start=1):
                    try:
                        simulator_state = _ensure_simulator_running(simulator_state)
                        _print_experiment_progress(
                            experiment_id=experiment_id,
                            experiment_index=experiment_index,
                            experiment_total=experiment_total,
                            scenario_id=scenario_id,
                            scenario_index=scenario_index,
                            scenario_total=scenario_total,
                            method_id=method_id,
                            method_index=method_index,
                            method_total=method_total,
                            stage="run-method",
                            task_count=experiment_summaries[experiment_id]["built_scenarios"][scenario_id]["task_count"],
                        )
                        if method_id not in methods_registry:
                            raise ValueError(f"Method {method_id} referenced by {experiment_id} is not defined in methods.json")
                        method_meta = methods_registry[method_id]
                        if str(method_meta.get("implementation_status") or "").strip().lower() != "ready":
                            raise RuntimeError(
                                f"Method {method_id} is not runnable. Reason: {method_meta.get('notes', '')}"
                            )
                        run_method_cmd = [
                                str(PYTHON_EXE),
                                str(RUN_METHOD_SCRIPT),
                                "--method",
                                method_id,
                                "--experiment",
                                experiment_id,
                                "--scenario",
                                scenario_id,
                                "--snapshot-id",
                                str(simulator_state["live_graph_snapshot_id"]),
                                "--start-index",
                                str(start_index),
                            ]
                        if args.use_deepseek:
                            run_method_cmd.append("--deepseek")
                        stdout = _run_command(
                            run_method_cmd,
                            label=f"run {experiment_id}/{scenario_id}/{method_id}",
                            stream_output=True,
                        )
                        payload = _parse_run_method_output(stdout)
                        run_record = {
                            "experiment_id": experiment_id,
                            "scenario_id": scenario_id,
                            "method_id": method_id,
                            "task_count": experiment_summaries[experiment_id]["built_scenarios"][scenario_id]["task_count"],
                            "result_path": str(payload["result_path"]),
                            "summary_path": str(payload["summary_path"]),
                        }
                        launched_runs.append(run_record)
                        experiment_summaries[experiment_id]["successful_runs"].append(run_record)

                        if args.skip_aggregate:
                            continue

                        _print_experiment_progress(
                            experiment_id=experiment_id,
                            experiment_index=experiment_index,
                            experiment_total=experiment_total,
                            scenario_id=scenario_id,
                            scenario_index=scenario_index,
                            scenario_total=scenario_total,
                            method_id=method_id,
                            method_index=method_index,
                            method_total=method_total,
                            stage="aggregate",
                            task_count=run_record["task_count"],
                        )
                        _run_command(
                            [
                                str(PYTHON_EXE),
                                str(AGGREGATE_SCRIPT),
                                "--result-jsonl",
                                str(payload["result_path"]),
                                "--method",
                                method_id,
                                "--experiment",
                                experiment_id,
                                "--scenario",
                                scenario_id,
                            ],
                            label=f"aggregate {experiment_id}/{scenario_id}/{method_id}",
                        )
                    except Exception as exc:
                        failure = {
                            "stage": "run_or_aggregate",
                            "experiment_id": experiment_id,
                            "scenario_id": scenario_id,
                            "method_id": method_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                        failed_runs.append(failure)
                        experiment_summaries[experiment_id]["failed_runs"].append(failure)
                        _append_failed_run(failure)
                        print(
                            f"[launch][skip] failed {experiment_id}/{scenario_id}/{method_id}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
            finally:
                _stop_simulator(simulator_state, reason=f"{experiment_id}/{scenario_id} complete")

    print(
        json.dumps(
            {
                "launched_run_count": len(launched_runs),
                "launched_runs": launched_runs,
                "failed_run_count": len(failed_runs),
                "failed_runs_path": str(FAILED_RUNS_PATH),
                "aggregation_skipped": bool(args.skip_aggregate),
                "coverage_summary": _build_coverage_summary(experiment_summaries),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
