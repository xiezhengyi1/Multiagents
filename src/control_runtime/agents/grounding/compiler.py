from __future__ import annotations

from typing import Dict, List, Optional

from ...context.evidence import EvidenceFormatter
from ...domain.policy_plan import OperationIntent
from .artifact_compiler import OperationIntentCompiler
from .common import (
    AM_GROUNDING_TOOLS,
    SM_GROUNDING_TOOLS,
    VALID_DOMAINS,
    MainDirectiveExtractor,
    uses_am_grounding,
    uses_sm_grounding,
)
from .contracts import IntentAdvisorDecision, IntentEvidence
from .validator import IntentGroundingValidator


class IntentCompiler:
    VALID_DOMAINS = VALID_DOMAINS
    SM_GROUNDING_TOOLS = SM_GROUNDING_TOOLS
    AM_GROUNDING_TOOLS = AM_GROUNDING_TOOLS

    def __init__(self) -> None:
        self.directive_extractor = MainDirectiveExtractor()
        self.validator = IntentGroundingValidator()
        self.operation_compiler = OperationIntentCompiler()

    @classmethod
    def uses_sm_grounding(cls, requested_domains: List[str] | None) -> bool:
        return uses_sm_grounding(requested_domains)

    @classmethod
    def uses_am_grounding(cls, requested_domains: List[str] | None) -> bool:
        return uses_am_grounding(requested_domains)

    def extract_main_directives(self, context: str) -> Dict[str, Any]:
        return self.directive_extractor.extract_main_directives(context)

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
        decision: IntentAdvisorDecision | None = None,
    ) -> List[str]:
        return self.validator.validate_intent_grounding(
            evidence=evidence,
            grounding_tools=grounding_tools,
            decision=decision,
        )

    def validate_advisor_decision(
        self,
        *,
        evidence: IntentEvidence,
        decision: IntentAdvisorDecision,
    ) -> List[str]:
        return self.validator.validate_advisor_decision(
            evidence=evidence,
            decision=decision,
        )

    def compile_operation_intent(
        self,
        *,
        evidence: IntentEvidence,
        advisor_decision: Optional[IntentAdvisorDecision],
        user_input: str,
        session_id: str,
        snapshot_id: str,
        main_directives: Optional[Dict[str, Any]] = None,
    ) -> OperationIntent:
        return self.operation_compiler.compile_operation_intent(
            evidence=evidence,
            advisor_decision=advisor_decision,
            user_input=user_input,
            session_id=session_id,
            snapshot_id=snapshot_id,
            main_directives=main_directives,
        )
