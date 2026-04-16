from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_runtime.io.files import ensure_directory
from agent_runtime.io.paths import agent_workspace_root, project_root, runtime_root


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
        root = agent_workspace_root(normalized_name)
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
        ensure_directory(self.cache_dir)
        ensure_directory(self.work_dir)
        ensure_directory(self.logs_dir)
