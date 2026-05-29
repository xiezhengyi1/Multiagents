from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AgentRuntimeContext:
    agent_name: str
    session_id: str
    snapshot_id: str
    supi: Optional[str]
    thread_id: str
    allow_user_interaction: bool = False
    token_budget: Optional[Any] = field(default=None, compare=False, hash=False)
    token_counter: Optional[Any] = field(default=None, compare=False, hash=False)
    trace_metadata: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


__all__ = ["AgentRuntimeContext"]
