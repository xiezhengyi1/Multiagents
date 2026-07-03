from __future__ import annotations

from typing import Any, List, Tuple

from ...domain.policy_plan import OperationIntent


def validate_operation_intent(
    *,
    compiler: Any,
    evidence: Any,
    operation_intent: OperationIntent,
    grounding_tools: List[str],
) -> Tuple[List[str], List[str], OperationIntent | None]:
    intent_errors = compiler.validate_operation_intent(
        evidence=evidence,
        operation_intent=operation_intent,
    )
    grounding_errors = compiler.validate_intent_grounding(
        evidence=evidence,
        grounding_tools=grounding_tools,
        operation_intent=operation_intent,
    )
    if intent_errors or grounding_errors:
        return list(intent_errors), list(grounding_errors), None
    return [], [], operation_intent
