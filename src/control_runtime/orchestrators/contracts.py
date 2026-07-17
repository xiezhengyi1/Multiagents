"""Artifacts exchanged between orchestration rounds and executors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ControlRoundTrace:
    round_index: int
    global_intent: Dict[str, Any] = field(default_factory=dict)
    grounding_decision: Dict[str, Any] = field(default_factory=dict)
    policy_plan: Dict[str, Any] = field(default_factory=dict)
    domain_verdicts: List[Dict[str, Any]] = field(default_factory=list)
    pda_feedback: Dict[str, Any] = field(default_factory=dict)
    qos_feedback: Dict[str, Any] = field(default_factory=dict)
    mobility_feedback: Dict[str, Any] = field(default_factory=dict)
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    negotiation_request: Dict[str, Any] = field(default_factory=dict)
    planning_blocker: Dict[str, Any] = field(default_factory=dict)
    execution_reentry: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlRoundResult:
    session_id: str
    snapshot_id: str
    completed: bool
    global_intent: Dict[str, Any]
    unified_plan: Dict[str, Any]
    qos_feedback: Dict[str, Any] = field(default_factory=dict)
    mobility_feedback: Dict[str, Any] = field(default_factory=dict)
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    negotiation_request: Dict[str, Any] = field(default_factory=dict)
    planning_blocker: Dict[str, Any] = field(default_factory=dict)
    execution_reentry: Dict[str, Any] = field(default_factory=dict)
    round_count: int = 1
    retry_count: int = 0
    round_traces: List[Dict[str, Any]] = field(default_factory=list)
    agent_elapsed_ms: Dict[str, float] = field(default_factory=dict)


__all__ = ["ControlRoundResult", "ControlRoundTrace"]
