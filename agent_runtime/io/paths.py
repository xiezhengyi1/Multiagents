from __future__ import annotations

from pathlib import Path


def normalize_name(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def runtime_root() -> Path:
    return project_root() / "runtime"


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


def trace_root(agent_name: str) -> Path:
    return project_root() / "training" / normalize_name(agent_name, field_name="agent_name") / "raw_traces"


__all__ = [
    "agent_workspace_root",
    "interface_requests_dir",
    "interface_responses_dir",
    "interface_root",
    "normalize_name",
    "project_root",
    "queue_root",
    "runtime_agents_root",
    "runtime_interfaces_root",
    "runtime_root",
    "trace_root",
]
