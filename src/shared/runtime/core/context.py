from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from shared.contracts import ExecutionScope, JsonValue


@dataclass(frozen=True)
class AgentRuntimeContext:
    agent_name: str
    scope: ExecutionScope
    allow_user_interaction: bool = False
    token_budget: Optional[Any] = field(default=None, compare=False, hash=False)
    token_counter: Optional[Any] = field(default=None, compare=False, hash=False)
    trace_attributes: Dict[str, JsonValue] = field(default_factory=dict, compare=False, hash=False)


__all__ = ["AgentRuntimeContext"]
