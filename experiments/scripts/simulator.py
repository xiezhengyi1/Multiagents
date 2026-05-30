from __future__ import annotations

from datetime import datetime
import http.client
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from experiments.paths import resolve_scenario_source_path
from experiments.scripts.common import (
    LEDGER_ROOT,
    WORKSPACE_ROOT,
    load_json,
    load_yaml_mapping,
    resolve_python_executable,
)


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


def run_command(
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
        output_text = "".join(output_lines).strip()
        return_code = process.wait()
        if return_code:
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
    stderr = completed.stderr.strip()
    if emit_output and stdout:
        print(stdout, flush=True)
    if emit_output and stderr:
        print(stderr, file=sys.stderr, flush=True)
    return stdout


def _resolve_scenario_path(scenario_id: str) -> Path:
    scenario_path = resolve_scenario_source_path(scenario_id)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario source not found for {scenario_id}: {scenario_path}")
    return scenario_path


def _resolve_simulator_scenario_path(scenario_id: str) -> Path:
    base_scenario_path = _resolve_scenario_path(scenario_id)
    split_mode_path = NS3_INTEGRATION_ROOT / "scenarios" / "split_mode" / base_scenario_path.name
    return split_mode_path if split_mode_path.exists() else base_scenario_path


def _resolve_graph_db_url(scenario_path: Path) -> str:
    payload = load_yaml_mapping(scenario_path)
    writer = payload.get("writer") if isinstance(payload.get("writer"), dict) else {}
    topology = payload.get("topology") if isinstance(payload.get("topology"), dict) else {}
    graph_db_url = str(writer.get("graph_db_url") or topology.get("graph_db_url") or "").strip()
    if graph_db_url:
        return graph_db_url
    base_scenario = payload.get("base_scenario")
    if isinstance(base_scenario, str) and base_scenario.strip():
        return _resolve_graph_db_url((scenario_path.parent / base_scenario).resolve())
    raise ValueError(f"Scenario YAML does not define writer.graph_db_url or topology.graph_db_url: {scenario_path}")


def _resolve_simulator_manifest_path(run_dir: Path) -> Path:
    split_manifest_path = run_dir / "run-manifest.split.json"
    if split_manifest_path.exists() or START_SIMULATOR_SCRIPT.name == "start_split_mode.py":
        return split_manifest_path
    return run_dir / "run-manifest.json"


def _build_experiment_live_graph_snapshot_id(experiment_id: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in str(experiment_id or "").strip())
    return f"live-{normalized.strip('-') or 'experiment'}"


def _run_graph_admin_command(arguments: List[str], *, label: str, emit_output: bool = True) -> Dict[str, Any]:
    stdout = run_command(
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
        ["graph-snapshot-exists", "--graph-db-url", graph_db_url, "--snapshot-id", snapshot_id, "--ensure-graph-schema"],
        label=f"check graph snapshot {snapshot_id}",
        emit_output=False,
    )
    return bool(payload.get("exists"))


def _delete_graph_snapshot(*, graph_db_url: str, snapshot_id: str) -> Dict[str, Any]:
    return _run_graph_admin_command(
        ["delete-graph-snapshot", "--graph-db-url", graph_db_url, "--snapshot-id", snapshot_id, "--ensure-graph-schema"],
        label=f"delete graph snapshot {snapshot_id}",
    )


def _prune_graph_snapshots(*, graph_db_url: str, keep_snapshot_id: str = "", keep_latest: bool = False) -> Dict[str, Any]:
    arguments = ["prune-graph-snapshots", "--graph-db-url", graph_db_url, "--ensure-graph-schema"]
    if keep_snapshot_id:
        arguments.extend(["--keep-snapshot-id", keep_snapshot_id])
    if keep_latest:
        arguments.append("--keep-latest")
    return _run_graph_admin_command(
        arguments,
        label=f"prune graph snapshots keep {keep_snapshot_id}" if keep_snapshot_id else "prune graph snapshots keep latest",
    )


def _read_log_tail(path: Path, *, max_lines: int = 20) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines:])


def _mirror_log_increment(*, source_path: Path, destination_path: Path, cursor: int = 0) -> int:
    if not source_path.exists():
        return cursor
    try:
        with source_path.open("r", encoding="utf-8", errors="ignore") as source:
            source.seek(cursor)
            chunk = source.read()
            cursor = source.tell()
    except OSError:
        return cursor
    if chunk:
        with destination_path.open("a", encoding="utf-8") as destination:
            destination.write(chunk)
    return cursor


def _run_compose_down_from_manifest(manifest_path: Path) -> None:
    if not manifest_path.exists():
        return
    commands = load_json(manifest_path).get("commands", [])
    compose_down = next((item for item in commands if str(item.get("name") or "") == "compose-down"), None)
    if not isinstance(compose_down, dict) or not compose_down.get("argv"):
        return
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in (compose_down.get("env") or {}).items()})
    subprocess.run(
        [str(item) for item in compose_down["argv"]],
        cwd=str(compose_down.get("cwd") or NS3_INTEGRATION_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _probe_simulator_gateway(*, timeout_sec: float = SIMULATOR_GATEWAY_TIMEOUT_SEC) -> tuple[bool, str]:
    connection = http.client.HTTPConnection(SIMULATOR_GATEWAY_HOST, SIMULATOR_GATEWAY_PORT, timeout=timeout_sec)
    try:
        connection.request("GET", SIMULATOR_GATEWAY_PROBE_PATH)
        response = connection.getresponse()
        status_code = int(response.status)
        detail = f"http {status_code}{(' ' + response.reason) if response.reason else ''}"
        response.read()
        return (200 <= status_code < 500), detail
    except Exception as exc:
        return False, str(exc)
    finally:
        connection.close()


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
    acceptor_log = manifest_path.parent / "logs" / "policy-acceptor.log"
    cursor = 0
    while time.monotonic() < deadline:
        cursor = _mirror_log_increment(source_path=acceptor_log, destination_path=log_path, cursor=cursor)
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
    _mirror_log_increment(source_path=acceptor_log, destination_path=log_path, cursor=cursor)
    raise RuntimeError(
        "Simulator bootstrap did not become ready "
        f"(snapshot_id={live_graph_snapshot_id}, rc={process.poll()}, gateway={last_gateway_detail}, "
        f"manifest={manifest_path}, log={log_path})\n{_read_log_tail(log_path, max_lines=40)}"
    )


def start_simulator_for_scenario(
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
    snapshot_id = _build_experiment_live_graph_snapshot_id(experiment_id)
    if reset_live_graph:
        _prune_graph_snapshots(graph_db_url=graph_db_url, keep_latest=True)
        _delete_graph_snapshot(graph_db_url=graph_db_url, snapshot_id=snapshot_id)
    run_id = f"batch-{experiment_id.lower()}-{scenario_id.lower()}-{datetime.now():%Y%m%d-%H%M%S}"
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
        snapshot_id,
        "--wait-background",
    ]
    print(f"[launch] start simulator for {experiment_id}/{scenario_id}: {scenario_path} (live_graph_snapshot_id={snapshot_id})", flush=True)
    process = subprocess.Popen(command, cwd=NS3_INTEGRATION_ROOT, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    try:
        _wait_for_simulator_bootstrap(
            process=process,
            manifest_path=manifest_path,
            log_path=log_path,
            graph_db_url=graph_db_url,
            live_graph_snapshot_id=snapshot_id,
        )
        _prune_graph_snapshots(graph_db_url=graph_db_url, keep_snapshot_id=snapshot_id)
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
        "live_graph_snapshot_id": snapshot_id,
    }


def restart_simulator_for_current_scenario(state: Dict[str, Any], *, reason: str) -> Dict[str, Any]:
    restart_count = int(state.get("restart_count", 0) or 0) + 1
    print(f"[launch] restart simulator for {state['experiment_id']}/{state['scenario_id']} (attempt={restart_count}, reason={reason})", flush=True)
    stop_simulator(state, reason=f"restart requested: {reason}")
    restarted = start_simulator_for_scenario(state["experiment_id"], state["scenario_id"], reset_live_graph=False)
    restarted["restart_count"] = restart_count
    return restarted


def ensure_simulator_running(state: Dict[str, Any]) -> Dict[str, Any]:
    rc = state["process"].poll()
    if rc is None:
        gateway_ready, detail = _probe_simulator_gateway()
        if gateway_ready:
            return state
        failure = f"policy gateway unhealthy for {state['scenario_id']} (run_id={state['run_id']}, live_graph_snapshot_id={state['live_graph_snapshot_id']}, gateway={detail})"
    else:
        failure = f"Simulator exited early for {state['scenario_id']} (run_id={state['run_id']}, live_graph_snapshot_id={state['live_graph_snapshot_id']}, rc={rc})"
    if int(state.get("restart_count", 0) or 0) < SIMULATOR_RESTART_LIMIT:
        return restart_simulator_for_current_scenario(state, reason=failure)
    raise RuntimeError(f"{failure}, log={state['log_path']}\n{_read_log_tail(state['log_path'], max_lines=40)}")


def stop_simulator(state: Dict[str, Any] | None, *, reason: str) -> None:
    if not state:
        return
    process = state["process"]
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
        state["log_handle"].close()
        if not graceful:
            _run_compose_down_from_manifest(state["manifest_path"])
