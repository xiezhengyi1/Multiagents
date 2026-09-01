from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import Field, model_validator

from shared.contracts import ContractModel, ExecutionScope, NonEmptyStr

from ...contracts import ControlDomain, ObjectiveProfile


class MainRoundStrategy(str, Enum):
    INITIAL_GROUNDING = "initial_grounding"
    REGROUNDING = "regrounding"
    POLICY_REVISION = "policy_revision"
    JOINT_REPLAN = "joint_replan"


class MainInvestigationTarget(str, Enum):
    DOMAIN_BOUNDARY = "domain_boundary"
    UE_BINDING = "ue_binding"
    QOS_FLOW_BINDING = "qos_flow_binding"
    MOBILITY_TARGET_BINDING = "mobility_target_binding"
    POLICY_FEASIBILITY = "policy_feasibility"
    CROSS_DOMAIN_CONSISTENCY = "cross_domain_consistency"
    ASSURANCE_GAP = "assurance_gap"


class MainUncertaintyFlag(str, Enum):
    DOMAIN_AMBIGUOUS = "domain_ambiguous"
    IDENTIFIER_RISK = "identifier_risk"
    RUNTIME_EVIDENCE_MISSING = "runtime_evidence_missing"
    EXECUTION_FEEDBACK_INCOMPLETE = "execution_feedback_incomplete"
    CONFLICT_SIGNAL_PRESENT = "conflict_signal_present"


class MainRetryScope(str, Enum):
    FULL_REGROUND = "full_reground"
    PARTIAL_REGROUND = "partial_reground"
    TARGET_STABLE = "target_stable"
    EXECUTION_RETRY_FORBIDDEN = "execution_retry_forbidden"


class SemanticGoal(str, Enum):
    PROTECT = "protect"
    DEPRIORITIZE = "deprioritize"
    DEFER = "defer"
    OBSERVE = "observe"


class SemanticTargetType(str, Enum):
    FLOW = "flow"
    APP = "app"
    SCOPE = "scope"
    NAMED_OBJECT = "named_object"


class StageTrigger(str, Enum):
    INITIAL = "initial"
    ON_PREVIOUS_FAILURE = "on_previous_failure"
    AFTER_PREVIOUS_STAGE = "after_previous_stage"
    RETRY = "retry"


class ControlSemanticMode(str, Enum):
    SINGLE_STEP = "single_step"
    STAGED_PRIORITY = "staged_priority"
    CONDITIONAL_FALLBACK = "conditional_fallback"


class MainSemanticTarget(ContractModel):
    semantic_name: str
    target_type: SemanticTargetType
    goal: SemanticGoal
    metric_focus: Optional[str]
    note: str
    supi: str


class MainControlStage(ContractModel):
    stage_index: int = Field(ge=1)
    name: str
    trigger: StageTrigger
    summary: str
    targets: list[MainSemanticTarget]


class MainControlSemantics(ContractModel):
    mode: ControlSemanticMode
    current_stage: int = Field(ge=1)
    stages: list[MainControlStage]
    notes: list[NonEmptyStr]

    @model_validator(mode="after")
    def validate_stage_selection(self) -> "MainControlSemantics":
        if len(self.notes) != len(set(self.notes)):
            raise ValueError("control semantic notes must be unique")
        if not self.stages:
            if self.current_stage != 1:
                raise ValueError("current_stage must be 1 when stages is empty")
            return self
        stage_indices = [stage.stage_index for stage in self.stages]
        if len(stage_indices) != len(set(stage_indices)):
            raise ValueError("control semantic stage_index values must be unique")
        if self.current_stage not in set(stage_indices):
            raise ValueError("current_stage must reference an existing stage_index")
        return self


class ReuseContract(ContractModel):
    allowed: bool = False
    preserve_bindings: bool = False
    preserve_domains: bool = False
    preserve_stage_scope: bool = False
    invalidate_on: list[NonEmptyStr] = Field(default_factory=list)

class MainControlDecision(ContractModel):
    supi: str
    next_agent: Literal["intent_encoding", "optimization_strategy"]
    requested_domains: list[ControlDomain]
    serving_slice_change_requested: bool
    domain_evidence: dict[ControlDomain, list[NonEmptyStr]]
    control_semantics: MainControlSemantics
    objective_profile: ObjectiveProfile
    investigation_targets: list[MainInvestigationTarget]
    uncertainty_flags: list[MainUncertaintyFlag]
    retry_scope: Optional[MainRetryScope]
    required_evidence: list[NonEmptyStr]
    forbidden_assumptions: list[NonEmptyStr]
    intent_encoding_guidance: str

    @model_validator(mode="after")
    def validate_unique_collections(self) -> "MainControlDecision":
        for name, values in (
            ("requested_domains", self.requested_domains),
            ("investigation_targets", self.investigation_targets),
            ("uncertainty_flags", self.uncertainty_flags),
            ("required_evidence", self.required_evidence),
            ("forbidden_assumptions", self.forbidden_assumptions),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        return self


class OpenQuestion(ContractModel):
    owner_agent: str = Field(default="")
    question: str = Field(default="")
    blocking: bool = False
    related_domains: list[NonEmptyStr] = Field(default_factory=list)


class GlobalControlIntent(ContractModel):
    scope: ExecutionScope
    raw_input: str = ""
    round_strategy: MainRoundStrategy = Field(default=MainRoundStrategy.INITIAL_GROUNDING)
    next_agent: Literal["intent_encoding", "optimization_strategy"]
    requested_domains: list[ControlDomain] = Field(default_factory=list)
    serving_slice_change_requested: bool = False
    domain_evidence: dict[ControlDomain, list[NonEmptyStr]] = Field(default_factory=dict)
    control_semantics: MainControlSemantics = Field(
        default_factory=lambda: MainControlSemantics(
            mode=ControlSemanticMode.SINGLE_STEP,
            current_stage=1,
            stages=[],
            notes=[],
        )
    )
    objective_profile: ObjectiveProfile = Field(default_factory=ObjectiveProfile)
    investigation_targets: list[MainInvestigationTarget] = Field(default_factory=list)
    uncertainty_flags: list[MainUncertaintyFlag] = Field(default_factory=list)
    retry_scope: Optional[MainRetryScope] = Field(default=None)
    required_evidence: list[NonEmptyStr] = Field(default_factory=list)
    forbidden_assumptions: list[NonEmptyStr] = Field(default_factory=list)
    intent_encoding_guidance: str = ""
    routing_decision: str = Field(default="")
    routing_rationale: str = Field(default="")
    reuse_contract: ReuseContract = Field(default_factory=ReuseContract)


__all__ = [
    "ControlDomain",
    "ControlSemanticMode",
    "GlobalControlIntent",
    "MainControlDecision",
    "MainControlSemantics",
    "MainControlStage",
    "MainInvestigationTarget",
    "MainRetryScope",
    "MainRoundStrategy",
    "MainSemanticTarget",
    "MainUncertaintyFlag",
    "ObjectiveProfile",
    "OpenQuestion",
    "ReuseContract",
    "SemanticGoal",
    "SemanticTargetType",
    "StageTrigger",
]
