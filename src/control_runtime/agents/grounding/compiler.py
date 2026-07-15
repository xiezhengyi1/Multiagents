from __future__ import annotations

from typing import Dict, List, Optional, Any

from ...context.evidence import EvidenceFormatter
from ...context.iea_contract import (
    AM_GROUNDING_TOOLS,
    SM_GROUNDING_TOOLS,
    VALID_DOMAINS,
    IntentEncodingDirectives,
    uses_am_grounding,
    uses_sm_grounding,
)
from ...domain.policy_plan import OperationIntent
from .contracts import IntentEvidence
from .validator import IntentGroundingValidator


class IntentCompiler:
    VALID_DOMAINS = VALID_DOMAINS
    SM_GROUNDING_TOOLS = SM_GROUNDING_TOOLS
    AM_GROUNDING_TOOLS = AM_GROUNDING_TOOLS

    def __init__(self) -> None:
        self.validator = IntentGroundingValidator()

    @classmethod
    def uses_sm_grounding(cls, requested_domains: List[str] | None) -> bool:
        return uses_sm_grounding(requested_domains)

    @classmethod
    def uses_am_grounding(cls, requested_domains: List[str] | None) -> bool:
        return uses_am_grounding(requested_domains)

    def extract_main_directives(self, context: str) -> Dict[str, Any]:
        return IntentEncodingDirectives.from_context(context).model_dump()

    def build_intent_evidence(
        self,
        *,
        user_input: str,
        supi: str,
        main_directives: Dict[str, Any],
        catalog_payload: Dict[str, Any],
        semantic_candidates: List[Dict[str, Any]],
        am_context_payload: Optional[Dict[str, Any]] = None,
        am_policy_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> IntentEvidence:
        return EvidenceFormatter.for_iea(
            user_input=user_input,
            supi=supi,
            main_directives=main_directives,
            catalog_payload=catalog_payload,
            semantic_candidates=semantic_candidates,
            am_context_payload=am_context_payload,
            am_policy_candidates=am_policy_candidates,
        )

    def validate_intent_grounding(
        self,
        *,
        evidence: IntentEvidence,
        grounding_tools: List[str],
        operation_intent: OperationIntent | None = None,
    ) -> List[str]:
        return self.validator.validate_intent_grounding(
            evidence=evidence,
            grounding_tools=grounding_tools,
            operation_intent=operation_intent,
        )

    def validate_operation_intent(
        self,
        *,
        evidence: IntentEvidence,
        operation_intent: OperationIntent,
    ) -> List[str]:
        return self.validator.validate_operation_intent(
            evidence=evidence,
            operation_intent=operation_intent,
        )
