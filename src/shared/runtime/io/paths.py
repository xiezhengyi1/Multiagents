from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re


TRACE_RUN_ID_ENV = "MULTIAGENTS_TRACE_RUN_ID"
TRACE_WORKER_ID_ENV = "MULTIAGENTS_TRACE_WORKER_ID"
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PROCESS_TRACE_RUN_ID = (
    "runtime-"
    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    + f"-{os.getpid()}"
)
_PROCESS_TRACE_WORKER_ID = f"process-{os.getpid()}"


def normalize_name(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def normalize_path_component(value: str, *, field_name: str) -> str:
    normalized = normalize_name(value, field_name=field_name)
    if _PATH_COMPONENT.fullmatch(normalized) is None:
        raise ValueError(
            f"{field_name} must contain only letters, digits, dots, underscores, and hyphens"
        )
    return normalized


def project_root() -> Path:
    for directory in Path(__file__).resolve().parents:
        if (directory / "pyproject.toml").is_file():
            return directory
    raise RuntimeError("Unable to locate repository root from runtime module path")


def runtime_root() -> Path:
    return project_root() / "data" / "runtime"


def runtime_agents_root() -> Path:
    return runtime_root() / "agents"


def agent_workspace_root(agent_name: str) -> Path:
    return runtime_agents_root() / normalize_name(agent_name, field_name="agent_name")


def runtime_interfaces_root() -> Path:
    return runtime_root() / "interfaces"


def interface_root(source_agent: str, target_agent: str) -> Path:
    source = normalize_name(source_agent, field_name="source_agent")
    target = normalize_name(target_agent, field_name="target_agent")
    return runtime_interfaces_root() / f"{source}__{target}"


def interface_requests_dir(source_agent: str, target_agent: str) -> Path:
    return interface_root(source_agent, target_agent) / "requests"


def interface_responses_dir(source_agent: str, target_agent: str) -> Path:
    return interface_root(source_agent, target_agent) / "responses"


def queue_root(agent_name: str) -> Path:
    return runtime_root() / "queues" / normalize_name(agent_name, field_name="agent_name")


def trace_runs_root() -> Path:
    return project_root() / "data" / "traces" / "runs"


def current_trace_identity() -> tuple[str, str]:
    configured_run_id = str(os.getenv(TRACE_RUN_ID_ENV, "")).strip()
    configured_worker_id = str(os.getenv(TRACE_WORKER_ID_ENV, "")).strip()
    if bool(configured_run_id) != bool(configured_worker_id):
        raise RuntimeError(
            f"{TRACE_RUN_ID_ENV} and {TRACE_WORKER_ID_ENV} must be configured together"
        )
    run_id = configured_run_id or _PROCESS_TRACE_RUN_ID
    worker_id = configured_worker_id or _PROCESS_TRACE_WORKER_ID
    return (
        normalize_path_component(run_id, field_name="dataset_run_id"),
        normalize_path_component(worker_id, field_name="worker_id"),
    )


def trace_run_root(dataset_run_id: str, *, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else trace_runs_root()
    return base / normalize_path_component(dataset_run_id, field_name="dataset_run_id")


def trace_worker_root(
    dataset_run_id: str,
    worker_id: str,
    *,
    root: Path | None = None,
) -> Path:
    return trace_run_root(dataset_run_id, root=root) / "workers" / normalize_path_component(
        worker_id,
        field_name="worker_id",
    )


def trace_segment_path(
    agent_name: str,
    *,
    dataset_run_id: str,
    worker_id: str,
    root: Path | None = None,
) -> Path:
    agent = normalize_path_component(agent_name, field_name="agent_name")
    return trace_worker_root(dataset_run_id, worker_id, root=root) / agent / f"{agent}.jsonl"


def current_trace_segment_path(agent_name: str, *, root: Path | None = None) -> Path:
    dataset_run_id, worker_id = current_trace_identity()
    return trace_segment_path(
        agent_name,
        dataset_run_id=dataset_run_id,
        worker_id=worker_id,
        root=root,
    )


__all__ = [
    "agent_workspace_root",
    "interface_requests_dir",
    "interface_responses_dir",
    "interface_root",
    "normalize_name",
    "normalize_path_component",
    "project_root",
    "queue_root",
    "runtime_agents_root",
    "runtime_interfaces_root",
    "runtime_root",
    "current_trace_identity",
    "current_trace_segment_path",
    "trace_run_root",
    "trace_runs_root",
    "trace_segment_path",
    "trace_worker_root",
]
