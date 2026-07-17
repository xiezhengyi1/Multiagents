from __future__ import annotations

from typing import Any, List, Tuple

from ...domain.policy_plan import GroundingDecision


def validate_grounding_decision(
    *,
    compiler: Any,
    evidence: Any,
    grounding_decision: GroundingDecision,
    grounding_tools: List[str],
) -> Tuple[List[str], List[str], GroundingDecision | None]:
    decision_errors = compiler.validate_grounding_decision(
        evidence=evidence,
        grounding_decision=grounding_decision,
    )
    grounding_errors = compiler.validate_intent_grounding(
        evidence=evidence,
        grounding_tools=grounding_tools,
        grounding_decision=grounding_decision,
    )
    if decision_errors or grounding_errors:
        return list(decision_errors), list(grounding_errors), None
    return [], [], grounding_decision
