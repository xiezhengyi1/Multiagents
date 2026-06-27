from __future__ import annotations

from typing import Any, Dict, List

from ...context.evidence import EvidenceFormatter, build_slice_snssai
from ...domain.collaboration import PlanningRequest
from ...domain.policy_plan import PolicyPlanDraft
from .planning_artifact_compiler import PlanningArtifactCompiler
from .planning_validation import PlanningAdvisorValidator, PlanningArtifactValidator
from .response_models import OsaAdvisorOutput


class OptimizationStrategyCompiler:
    def __init__(self) -> None:
        self.advisor_validator = PlanningAdvisorValidator()
        self.plan_validator = PlanningArtifactValidator()
        self.artifact_compiler = PlanningArtifactCompiler(validator=self.plan_validator)

    def build_planning_evidence(self, planning_request: PlanningRequest) -> Dict[str, Any]:
        return EvidenceFormatter.for_osa(
            operation_intent=planning_request.operation_intent,
            planning_context=planning_request.context,
        )

    def validate_advisor_output(
        self,
        *,
        advisor_output: OsaAdvisorOutput,
        planning_request: PlanningRequest,
        grounding_tools: List[str],
        planning_tool_evidence: Dict[str, Any] | None = None,
    ) -> List[str]:
        return self.advisor_validator.validate_advisor_output(
            advisor_output=advisor_output,
            planning_request=planning_request,
            grounding_tools=grounding_tools,
            planning_tool_evidence=planning_tool_evidence,
        )

    def assemble_policy_plan(
        self,
        *,
        advisor_output: OsaAdvisorOutput,
        planning_request: PlanningRequest,
        planning_tool_evidence: Dict[str, Any],
    ) -> PolicyPlanDraft:
        return self.artifact_compiler.assemble_policy_plan(
            advisor_output=advisor_output,
            planning_request=planning_request,
            planning_tool_evidence=planning_tool_evidence,
        )

    def build_upstream_reground_plan(
        self,
        *,
        planning_request: PlanningRequest,
        reason: str,
        advisor_output: OsaAdvisorOutput | None = None,
    ) -> PolicyPlanDraft:
        return self.artifact_compiler.build_upstream_reground_plan(
            planning_request=planning_request,
            reason=reason,
            advisor_output=advisor_output,
        )


__all__ = ["OptimizationStrategyCompiler", "build_slice_snssai"]
