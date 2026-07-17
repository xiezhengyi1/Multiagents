from __future__ import annotations

from typing import Dict, List, Optional, Any

from ...context.evidence import EvidenceFormatter
from ...domain.intent_encoding import (
    AM_GROUNDING_TOOLS,
    SM_GROUNDING_TOOLS,
    VALID_DOMAINS,
    IntentEncodingDirectives,
    uses_am_grounding,
    uses_sm_grounding,
)
from ...domain.policy_plan import GroundingDecision
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
        catalog_evidence_observed: bool = False,
        semantic_candidates: List[Dict[str, Any]],
        am_context_payload: Optional[Dict[str, Any]] = None,
        am_policy_candidates: Optional[List[Dict[str, Any]]] = None,
        subscription_payload: Optional[Dict[str, Any]] = None,
    ) -> IntentEvidence:
        return EvidenceFormatter.for_iea(
            user_input=user_input,
            supi=supi,
            main_directives=main_directives,
            catalog_payload=catalog_payload,
            catalog_evidence_observed=catalog_evidence_observed,
            semantic_candidates=semantic_candidates,
            am_context_payload=am_context_payload,
            am_policy_candidates=am_policy_candidates,
            subscription_payload=subscription_payload,
        )

    def validate_intent_grounding(
        self,
        *,
        evidence: IntentEvidence,
        grounding_tools: List[str],
        grounding_decision: GroundingDecision | None = None,
    ) -> List[str]:
        return self.validator.validate_intent_grounding(
            evidence=evidence,
            grounding_tools=grounding_tools,
            grounding_decision=grounding_decision,
        )

    def validate_grounding_decision(
        self,
        *,
        evidence: IntentEvidence,
        grounding_decision: GroundingDecision,
    ) -> List[str]:
        return self.validator.validate_grounding_decision(
            evidence=evidence,
            grounding_decision=grounding_decision,
        )
