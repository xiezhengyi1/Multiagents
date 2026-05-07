from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AgentRuntimeContext:
    agent_name: str
    session_id: str
    snapshot_id: str
    supi: Optional[str]
    thread_id: str
    allow_user_interaction: bool = False


__all__ = ["AgentRuntimeContext"]
