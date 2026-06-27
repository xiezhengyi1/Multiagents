from __future__ import annotations

from typing import Any, List, Tuple

from ...domain.policy_plan import OperationIntent


def validate_and_compile_intent(
    *,
    compiler: Any,
    evidence: Any,
    decision: Any,
    grounding_tools: List[str],
    user_input: str,
    session_id: str,
    snapshot_id: str,
    main_directives: dict[str, Any],
) -> Tuple[List[str], List[str], OperationIntent | None]:
    advisor_errors = compiler.validate_advisor_decision(
        evidence=evidence,
        decision=decision,
    )
    grounding_errors = compiler.validate_intent_grounding(
        evidence=evidence,
        grounding_tools=grounding_tools,
        decision=decision,
    )
    if advisor_errors or grounding_errors:
        return list(advisor_errors), list(grounding_errors), None
    compiled = compiler.compile_operation_intent(
        evidence=evidence,
        advisor_decision=decision,
        user_input=user_input,
        session_id=session_id,
        snapshot_id=snapshot_id,
        main_directives=main_directives,
    )
    return [], [], compiled
