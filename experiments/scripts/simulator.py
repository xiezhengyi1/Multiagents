from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from experiments.paths import resolve_scenario_source_path
from experiments.scripts.common import (
    LEDGER_ROOT,
    PROJECT_ROOT,
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
FAST_RESET_HOST = "127.0.0.1"
FAST_RESET_PORT = 18081
FAST_RESET_STATUS_PATH = "/v1/status"
FAST_RESET_TIMEOUT_SEC = 2.0
SIMULATOR_RESTART_LIMIT = 2
SIMULATOR_INSTANCE_SLOT_LIMIT = 2047
SIMULATOR_INSTANCE_PORT_BASE = 20000
SIMULATOR_INSTANCE_PORT_STRIDE = 4
SIMULATOR_INSTANCE_LOCK_ROOT = NS3_INTEGRATION_ROOT / "artifacts" / "parallel_slots"


@dataclass(slots=True)
class SimulatorInstanceAllocation:
    slot: int
    gateway_port: int
    reset_port: int
    instance_id: str
    database_schema: str
    lock_path: Path

    def release(self) -> None:
        try:
            self.lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def _slug(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return normalized or fallback


def _load_env_value(name: str) -> str:
    value = str(os.getenv(name) or "").strip()
    if value:
        return value
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return ""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, candidate = line.split("=", 1)
        if key.strip() == name:
            return candidate.strip().strip('"').strip("'")
    return ""


def _scoped_database_url(database_url: str, schema: str) -> str:
    parsed = urlsplit(str(database_url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError("DATABASE_URL must be configured before launching parallel simulator workers")
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["options"] = f"-csearch_path={schema},public"
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


def _ensure_database_schema(database_url: str, schema: str) -> None:
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:
        raise RuntimeError("SQLAlchemy is required to provision parallel worker schemas") from exc
    sqlalchemy_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_engine(sqlalchemy_url, pool_pre_ping=True)
    try:
        with engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
    finally:
        engine.dispose()


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((SIMULATOR_GATEWAY_HOST, port))
        except OSError:
            return False
    return True


def _lock_owner_is_alive(lock_path: Path) -> bool:
    """Return whether a slot lock belongs to a live local launcher process."""
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(payload.get("pid") or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def allocate_simulator_instance(*, label: str) -> SimulatorInstanceAllocation:
    """Reserve a local worker slot and its two host ports until simulator teardown."""
    SIMULATOR_INSTANCE_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    label_slug = _slug(label, fallback="experiment")
    for slot in range(1, SIMULATOR_INSTANCE_SLOT_LIMIT + 1):
        lock_path = SIMULATOR_INSTANCE_LOCK_ROOT / f"slot-{slot}.lock"
        if lock_path.exists() and not _lock_owner_is_alive(lock_path):
            print(f"[launch] reclaim stale simulator slot={slot} lock={lock_path}", flush=True)
            lock_path.unlink(missing_ok=True)
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        gateway_port = SIMULATOR_INSTANCE_PORT_BASE + ((slot - 1) * SIMULATOR_INSTANCE_PORT_STRIDE)
        reset_port = gateway_port + 1
        if not _port_is_available(gateway_port) or not _port_is_available(reset_port):
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)
            continue
        instance_id = f"slot-{slot}"
        database_schema = _slug(f"experiment-{label_slug}-slot-{slot}", fallback=f"experiment-slot-{slot}").replace("-", "_")[:63]
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "logical_label": label_slug,
                    "physical_slot": slot,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
                handle,
            )
        return SimulatorInstanceAllocation(
            slot=slot,
            gateway_port=gateway_port,
            reset_port=reset_port,
            instance_id=instance_id,
            database_schema=database_schema,
            lock_path=lock_path,
        )
    raise RuntimeError("No isolated simulator worker slot is available")


def run_command(
    command: List[str],
    *,
    label: str,
    cwd: Path = WORKSPACE_ROOT,
    emit_output: bool = True,
    stream_output: bool = False,
    env: Dict[str, str] | None = None,
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
            env=env,
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
        env=env,
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


def _build_experiment_live_graph_snapshot_id(experiment_id: str, *, instance_id: str = "") -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in str(experiment_id or "").strip())
    base = f"live-{normalized.strip('-') or 'experiment'}"
    instance = _slug(instance_id, fallback="")
    return f"{base}-{instance}" if instance else base


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


def _probe_simulator_gateway(
    *,
    port: int = SIMULATOR_GATEWAY_PORT,
    expected_instance_id: str = "",
    timeout_sec: float = SIMULATOR_GATEWAY_TIMEOUT_SEC,
) -> tuple[bool, str]:
    connection = http.client.HTTPConnection(SIMULATOR_GATEWAY_HOST, port, timeout=timeout_sec)
    try:
        connection.request("GET", SIMULATOR_GATEWAY_PROBE_PATH)
        response = connection.getresponse()
        status_code = int(response.status)
        raw_payload = response.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw_payload) if raw_payload else {}
        except json.JSONDecodeError:
            payload = {}
        is_policy_acceptor = (
            isinstance(payload, dict)
            and payload.get("service") == "ns3-free5gc-policy-acceptor"
            and payload.get("healthy") is True
            and (
                not expected_instance_id
                or payload.get("instance_id") == expected_instance_id
            )
        )
        detail = f"http {status_code}{(' ' + response.reason) if response.reason else ''}"
        if raw_payload:
            detail += f" body={raw_payload[:240]}"
        return status_code == 200 and is_policy_acceptor, detail
    except Exception as exc:
        return False, str(exc)
    finally:
        connection.close()


def _probe_fast_reset_service(
    *,
    port: int = FAST_RESET_PORT,
    timeout_sec: float = FAST_RESET_TIMEOUT_SEC,
) -> tuple[bool, str]:
    """Confirm that 18081 belongs to fast-reset, rather than merely being open."""
    connection = http.client.HTTPConnection(FAST_RESET_HOST, port, timeout=timeout_sec)
    try:
        connection.request("GET", FAST_RESET_STATUS_PATH)
        response = connection.getresponse()
        status_code = int(response.status)
        raw_payload = response.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw_payload) if raw_payload else {}
        except json.JSONDecodeError:
            payload = {}
        has_fast_reset_shape = isinstance(payload, dict) and any(
            key in payload for key in ("run_id", "initialized", "healthy")
        )
        detail = f"http {status_code}{(' ' + response.reason) if response.reason else ''}"
        if raw_payload:
            detail += f" body={raw_payload[:160]}"
        return status_code == 200 and has_fast_reset_shape, detail
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
    gateway_port: int,
    reset_port: int,
    expected_gateway_instance_id: str = "",
) -> None:
    deadline = time.monotonic() + SIMULATOR_BOOTSTRAP_TIMEOUT_SEC
    last_gateway_detail = "not checked"
    last_fast_reset_detail = "not checked"
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
        gateway_ready, last_gateway_detail = _probe_simulator_gateway(
            port=gateway_port,
            expected_instance_id=expected_gateway_instance_id,
        )
        fast_reset_ready, last_fast_reset_detail = _probe_fast_reset_service(port=reset_port)
        if graph_ready and gateway_ready and fast_reset_ready:
            return
        time.sleep(1.0)
    _mirror_log_increment(source_path=acceptor_log, destination_path=log_path, cursor=cursor)
    raise RuntimeError(
        "Simulator bootstrap did not become ready "
        f"(snapshot_id={live_graph_snapshot_id}, rc={process.poll()}, gateway={last_gateway_detail}, "
        f"fast_reset={last_fast_reset_detail}, "
        f"manifest={manifest_path}, log={log_path})\n{_read_log_tail(log_path, max_lines=40)}"
    )


def start_simulator_for_scenario(
    experiment_id: str,
    scenario_id: str,
    *,
    reset_live_graph: bool = True,
    instance: SimulatorInstanceAllocation | None = None,
) -> Dict[str, Any]:
    if not START_SIMULATOR_SCRIPT.exists():
        raise FileNotFoundError(f"Missing simulator launcher: {START_SIMULATOR_SCRIPT}")
    if not GRAPH_WRITER_CLI.exists():
        raise FileNotFoundError(f"Missing graph writer CLI: {GRAPH_WRITER_CLI}")
    scenario_path = _resolve_simulator_scenario_path(scenario_id)
    graph_db_url = _resolve_graph_db_url(scenario_path)
    runtime_database_url = ""
    if instance is not None:
        graph_db_url = _scoped_database_url(graph_db_url, instance.database_schema)
        runtime_database_url = _scoped_database_url(
            _load_env_value("DATABASE_URL"),
            instance.database_schema,
        )
        _ensure_database_schema(graph_db_url, instance.database_schema)
        if runtime_database_url != graph_db_url:
            _ensure_database_schema(runtime_database_url, instance.database_schema)
    snapshot_id = _build_experiment_live_graph_snapshot_id(
        experiment_id,
        instance_id=instance.instance_id if instance is not None else "",
    )
    if reset_live_graph:
        # Parallel workers have distinct snapshot ids but may share the same
        # backing graph store.  Pruning here would delete snapshots already
        # bootstrapped for earlier workers before their experiment process can
        # rebuild its scenario from them.
        if instance is None:
            _prune_graph_snapshots(graph_db_url=graph_db_url, keep_latest=True)
        _delete_graph_snapshot(graph_db_url=graph_db_url, snapshot_id=snapshot_id)
    run_id = (
        f"batch-{experiment_id.lower()}-{scenario_id.lower()}-"
        f"{instance.instance_id if instance is not None else 'default'}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    )
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
        "--gateway-port",
        str(instance.gateway_port if instance is not None else SIMULATOR_GATEWAY_PORT),
        "--reset-port",
        str(instance.reset_port if instance is not None else FAST_RESET_PORT),
        "--graph-db-url",
        graph_db_url,
        "--wait-background",
    ]
    if instance is not None:
        command.extend(["--instance-slot", str(instance.slot)])
    print(f"[launch] start simulator for {experiment_id}/{scenario_id}: {scenario_path} (live_graph_snapshot_id={snapshot_id})", flush=True)
    process = subprocess.Popen(command, cwd=NS3_INTEGRATION_ROOT, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    try:
        _wait_for_simulator_bootstrap(
            process=process,
            manifest_path=manifest_path,
            log_path=log_path,
            graph_db_url=graph_db_url,
            live_graph_snapshot_id=snapshot_id,
            gateway_port=instance.gateway_port if instance is not None else SIMULATOR_GATEWAY_PORT,
            reset_port=instance.reset_port if instance is not None else FAST_RESET_PORT,
            expected_gateway_instance_id=instance.instance_id if instance is not None else "default",
        )
        if instance is None:
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
        "instance": instance,
        "gateway_port": instance.gateway_port if instance is not None else SIMULATOR_GATEWAY_PORT,
        "gateway_instance_id": instance.instance_id if instance is not None else "default",
        "reset_port": instance.reset_port if instance is not None else FAST_RESET_PORT,
        "runtime_database_url": runtime_database_url,
    }


def restart_simulator_for_current_scenario(state: Dict[str, Any], *, reason: str) -> Dict[str, Any]:
    restart_count = int(state.get("restart_count", 0) or 0) + 1
    print(f"[launch] restart simulator for {state['experiment_id']}/{state['scenario_id']} (attempt={restart_count}, reason={reason})", flush=True)
    stop_simulator(state, reason=f"restart requested: {reason}", release_instance=False)
    restarted = start_simulator_for_scenario(
        state["experiment_id"],
        state["scenario_id"],
        reset_live_graph=False,
        instance=state.get("instance"),
    )
    restarted["restart_count"] = restart_count
    return restarted


def ensure_simulator_running(state: Dict[str, Any]) -> Dict[str, Any]:
    rc = state["process"].poll()
    if rc is None:
        gateway_ready, detail = _probe_simulator_gateway(
            port=int(state.get("gateway_port") or SIMULATOR_GATEWAY_PORT),
            expected_instance_id=str(state.get("gateway_instance_id") or ""),
        )
        if gateway_ready:
            return state
        failure = f"policy gateway unhealthy for {state['scenario_id']} (run_id={state['run_id']}, live_graph_snapshot_id={state['live_graph_snapshot_id']}, gateway={detail})"
    else:
        failure = f"Simulator exited early for {state['scenario_id']} (run_id={state['run_id']}, live_graph_snapshot_id={state['live_graph_snapshot_id']}, rc={rc})"
    if int(state.get("restart_count", 0) or 0) < SIMULATOR_RESTART_LIMIT:
        return restart_simulator_for_current_scenario(state, reason=failure)
    raise RuntimeError(f"{failure}, log={state['log_path']}\n{_read_log_tail(state['log_path'], max_lines=40)}")


def stop_simulator(
    state: Dict[str, Any] | None,
    *,
    reason: str,
    release_instance: bool = True,
) -> None:
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
        instance = state.get("instance")
        if release_instance and isinstance(instance, SimulatorInstanceAllocation):
            instance.release()
