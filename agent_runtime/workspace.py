from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def runtime_root() -> Path:
    return project_root() / "runtime"


@dataclass(frozen=True)
class AgentWorkspace:
    agent_name: str
    root: Path
    cache_dir: Path
    work_dir: Path
    logs_dir: Path

    @classmethod
    def for_agent(cls, agent_name: str) -> "AgentWorkspace":
        normalized_name = str(agent_name or "").strip()
        root = runtime_root() / "agents" / normalized_name
        workspace = cls(
            agent_name=normalized_name,
            root=root,
            cache_dir=root / "cache",
            work_dir=root / "work",
            logs_dir=root / "logs",
        )
        workspace.ensure()
        return workspace

    def ensure(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
