from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import yaml
import requests

from experiments.paths import (
    CONFIG_ROOT,
    GENERATED_INPUT_ROOT,
    LEDGER_ROOT,
    PACKAGE_ROOT as EXPERIMENT_ROOT,
    PROJECT_ROOT,
    RAW_RUN_ROOT,
    RESULTS_ROOT,
    SCENARIO_ROOT,
    SUMMARY_ROOT,
    TASK_ROOT,
    WORKSPACE_ROOT,
)


SIMULATOR_RESET_URL = str(os.getenv("SIMULATOR_RESET_URL") or "http://127.0.0.1:18081/v1/reset").strip()
SIMULATOR_RESET_TIMEOUT_SECONDS = 120.0
SIMULATOR_RESET_RETRY_INTERVAL_SECONDS = 1.0
SIMULATOR_RESET_REQUEST_TIMEOUT_SECONDS = 5.0
SIMULATOR_RESET_MAX_RECOVERY_REQUESTS = 3
SIMULATOR_RESET_RECOVERY_GRACE_SECONDS = 10.0
POLICY_GATEWAY_HEALTHCHECK_PATH = "/policy-executions/launch-healthcheck"


def _bounded_request_timeout(*, deadline: float) -> float:
    """Keep one unavailable endpoint from consuming the whole reset budget."""
    return min(
        SIMULATOR_RESET_REQUEST_TIMEOUT_SECONDS,
        max(0.01, deadline - time.monotonic()),
    )


def _response_body(response: Any) -> tuple[Dict[str, Any], str]:
    body = str(getattr(response, "text", "") or "").strip()
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        payload = {}
    return (dict(payload) if isinstance(payload, dict) else {}, body[:500])


def _reset_status_url(reset_url: str) -> str:
    parsed = urlsplit(str(reset_url or "").strip())
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/reset"):
        path = path[: -len("/reset")] + "/status"
    else:
        path = "/v1/status"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def _policy_gateway_health_url() -> str:
    base_url = str(os.getenv("PCF_BASE_URL") or "http://localhost:18080").strip().rstrip("/")
    return f"{base_url}{POLICY_GATEWAY_HEALTHCHECK_PATH}"


def _reset_status_is_complete(payload: Mapping[str, Any]) -> bool:
    progress = payload.get("reset_progress")
    return (
        bool(payload.get("healthy"))
        and isinstance(progress, Mapping)
        and progress.get("active") is False
        and str(progress.get("phase") or "").strip().lower() == "completed"
    )


def _missing_reset_processes(payload: Mapping[str, Any]) -> List[str]:
    """Return supervisor-declared missing runtime children, if any."""
    missing = payload.get("missing_processes")
    if not isinstance(missing, Sequence) or isinstance(missing, (str, bytes)):
        return []
    return [str(name or "").strip() for name in missing if str(name or "").strip()]


def _reset_is_active(payload: Mapping[str, Any]) -> bool:
    progress = payload.get("reset_progress")
    return isinstance(progress, Mapping) and progress.get("active") is True


def _policy_gateway_is_ready(payload: Mapping[str, Any]) -> bool:
    expected_instance_id = str(os.getenv("POLICY_GATEWAY_INSTANCE_ID") or "").strip()
    if expected_instance_id and str(payload.get("instance_id") or "").strip() != expected_instance_id:
        return False
    return (
        payload.get("healthy") is True
        and str(payload.get("service") or "").strip() == "ns3-free5gc-policy-acceptor"
    )


def reset_simulator_state(
    *,
    url: str = SIMULATOR_RESET_URL,
    timeout_seconds: float = SIMULATOR_RESET_TIMEOUT_SECONDS,
) -> None:
    deadline = time.monotonic() + max(0.01, float(timeout_seconds))
    reset_url = str(url or "").strip()
    status_url = _reset_status_url(reset_url)
    gateway_health_url = _policy_gateway_health_url()
    attempts = 0
    reset_accepted = False
    last_reset_status = 0
    last_reset_body = ""
    last_status_status = 0
    last_status_body = ""
    last_gateway_status = 0
    last_gateway_body = ""
    reset_accepted_at: float | None = None
    while True:
        status_payload: Dict[str, Any] = {}
        gateway_payload: Dict[str, Any] = {}
        if not reset_accepted:
            attempts += 1
            try:
                response = requests.post(reset_url, timeout=_bounded_request_timeout(deadline=deadline))
            except requests.RequestException as exc:
                last_reset_body = f"request error: {exc}"
            else:
                last_reset_status = int(response.status_code)
                reset_payload, last_reset_body = _response_body(response)
                if last_reset_status == 200:
                    reset_accepted = True
                    reset_accepted_at = time.monotonic()
                elif last_reset_status == 409:
                    # Another reset already owns the supervisor.  Its terminal
                    # status is sufficient for the next case; do not enqueue a
                    # second reset or race the same ns-3 instance.
                    reset_accepted = True
                    reset_accepted_at = time.monotonic()
                else:
                    is_policy_acceptor_404 = (
                        last_reset_status == 404
                        and str(reset_payload.get("status") or "").strip().lower() == "failed"
                        and str(reset_payload.get("error") or "").strip().lower() == "not found"
                    )
                    if is_policy_acceptor_404:
                        raise RuntimeError(
                            "Simulator reset endpoint identity mismatch: "
                            f"{reset_url} returned a policy-acceptor-style 404 response ({last_reset_body}). "
                            "Verify that the fast-reset service owns the reset port and exposes /v1/reset."
                        )
                    if not (500 <= last_reset_status < 600):
                        raise RuntimeError(
                            f"Simulator reset expected HTTP 200 or 409 but received {last_reset_status} after {attempts} attempts"
                            + (f": {last_reset_body}" if last_reset_body else "")
                        )

        try:
            status_response = requests.get(status_url, timeout=_bounded_request_timeout(deadline=deadline))
        except requests.RequestException as exc:
            last_status_body = f"request error: {exc}"
        else:
            last_status_status = int(status_response.status_code)
            status_payload, last_status_body = _response_body(status_response)
            progress = status_payload.get("reset_progress")
            if (
                isinstance(progress, Mapping)
                and progress.get("active") is False
                and str(progress.get("phase") or "").strip().lower() == "failed"
            ):
                raise RuntimeError(
                    "Simulator reset supervisor reported failure: "
                    + (str(progress.get("last_error") or last_status_body or "unknown error"))
                )

        try:
            gateway_response = requests.get(
                gateway_health_url,
                timeout=_bounded_request_timeout(deadline=deadline),
            )
        except requests.RequestException as exc:
            last_gateway_body = f"request error: {exc}"
            gateway_payload: Dict[str, Any] = {}
        else:
            last_gateway_status = int(gateway_response.status_code)
            gateway_payload, last_gateway_body = _response_body(gateway_response)

        if (
            reset_accepted
            and last_status_status == 200
            and _reset_status_is_complete(status_payload)
            and last_gateway_status == 200
            and _policy_gateway_is_ready(gateway_payload)
        ):
            print(
                f"[simulator] reset ready: reset=HTTP {last_reset_status or 409}, "
                f"status=HTTP {last_status_status}, gateway=HTTP {last_gateway_status}",
                flush=True,
            )
            return

        missing_processes = _missing_reset_processes(status_payload)
        recovery_grace_elapsed = (
            reset_accepted_at is not None
            and time.monotonic() - reset_accepted_at >= SIMULATOR_RESET_RECOVERY_GRACE_SECONDS
        )
        if (
            reset_accepted
            and last_status_status == 503
            and missing_processes
            and not _reset_is_active(status_payload)
            and recovery_grace_elapsed
            and attempts < SIMULATOR_RESET_MAX_RECOVERY_REQUESTS
        ):
            # A fast reset was accepted but its log-writer child exited before
            # readiness. Waiting cannot recreate that child. Re-submit a small,
            # bounded number of reset requests so the supervisor gets a chance
            # to rebuild the failed writer without letting a batch spin forever.
            print(
                "[simulator] reset recovery: supervisor reports missing process(es) "
                f"{','.join(missing_processes)}; requesting recovery reset "
                f"({attempts + 1}/{SIMULATOR_RESET_MAX_RECOVERY_REQUESTS})",
                flush=True,
            )
            reset_accepted = False

        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Simulator reset did not become ready within {timeout_seconds:.1f}s after {attempts} reset request(s). "
                f"reset=HTTP {last_reset_status or '<error>'} {last_reset_body}; "
                f"status=HTTP {last_status_status or '<error>'} {last_status_body}; "
                f"gateway=HTTP {last_gateway_status or '<error>'} {last_gateway_body}"
            )
        time.sleep(min(SIMULATOR_RESET_RETRY_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic())))



def resolve_python_executable(project_root: Path) -> Path:
    candidates = [
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "bin" / "python3",
        project_root / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable).resolve()


def parse_task_ids(value: str | Sequence[str] | None) -> List[str]:
    """Normalize a comma-separated task-id selector while preserving its order."""
    if value is None:
        return []
    raw_values = [value] if isinstance(value, str) else list(value)
    # argparse uses an empty string for an omitted optional selector in the
    # experiment entry points. Treat that default as no selector so callers
    # retain their normal start-index/count behavior.
    if not any(str(raw_value or "").strip() for raw_value in raw_values):
        return []
    task_ids: List[str] = []
    for raw_value in raw_values:
        for item in str(raw_value or "").split(","):
            task_id = item.strip()
            if not task_id:
                continue
            if task_id in task_ids:
                raise ValueError(f"--task-ids contains duplicate task_id={task_id}")
            task_ids.append(task_id)
    if raw_values and not task_ids:
        raise ValueError("--task-ids must contain at least one non-empty task id")
    return task_ids


def load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def build_project_python_env() -> Dict[str, str]:
    env = dict(os.environ)
    pythonpath = str(PROJECT_ROOT / "src")
    existing = str(env.get("PYTHONPATH") or "").strip()
    env["PYTHONPATH"] = pythonpath + ((os.pathsep + existing) if existing else "")
    return env


def run_project_python(script: Path, *args: str, cwd: Path = WORKSPACE_ROOT) -> None:
    subprocess.run(
        [str(resolve_python_executable(PROJECT_ROOT)), str(script), *args],
        cwd=cwd,
        env=build_project_python_env(),
        check=True,
    )


__all__ = [
    "CONFIG_ROOT",
    "EXPERIMENT_ROOT",
    "GENERATED_INPUT_ROOT",
    "LEDGER_ROOT",
    "PROJECT_ROOT",
    "RAW_RUN_ROOT",
    "RESULTS_ROOT",
    "SCENARIO_ROOT",
    "SUMMARY_ROOT",
    "TASK_ROOT",
    "WORKSPACE_ROOT",
    "append_jsonl",
    "build_project_python_env",
    "load_json",
    "load_yaml_mapping",
    "read_jsonl",
    "reset_simulator_state",
    "resolve_python_executable",
    "run_project_python",
    "parse_task_ids",
    "write_csv",
    "write_json",
]
