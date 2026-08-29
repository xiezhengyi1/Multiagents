from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class MainSemanticTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_name: str = Field(default="")
    target_type: SemanticTargetType = Field(default=SemanticTargetType.NAMED_OBJECT)
    goal: SemanticGoal = Field(default=SemanticGoal.PROTECT)
    metric_focus: Optional[str] = Field(default=None)
    note: str = Field(default="")
    supi: str = Field(default="")


class MainControlStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_index: int = Field(default=1, ge=1)
    name: str = Field(default="")
    trigger: StageTrigger = Field(default=StageTrigger.INITIAL)
    summary: str = Field(default="")
    targets: List[MainSemanticTarget] = Field(default_factory=list)


class MainControlSemantics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ControlSemanticMode = Field(default=ControlSemanticMode.SINGLE_STEP)
    current_stage: int = Field(default=1, ge=1)
    stages: List[MainControlStage] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_current_stage(self) -> "MainControlSemantics":
        if not self.stages:
            self.current_stage = 1
            return self
        self.current_stage = min(self.current_stage, max(stage.stage_index for stage in self.stages))
        return self


class ReuseContract(BaseModel):
    allowed: bool = False
    preserve_bindings: bool = False
    preserve_domains: bool = False
    preserve_stage_scope: bool = False
    invalidate_on: List[str] = Field(default_factory=list)


class MainControlDecisionWire(BaseModel):
    """Permissive transport shape repaired before strict decision validation."""

    model_config = ConfigDict(extra="allow")

    supi: str = ""
    next_agent: str = ""
    requested_domains: List[Any] = []
    serving_slice_change_requested: bool = False
    domain_evidence: Dict[str, Any] = {}
    control_semantics: Dict[str, Any] = {}
    objective_profile: Dict[str, Any] = {}
    investigation_targets: List[Any] = []
    uncertainty_flags: List[Any] = []
    retry_scope: str = ""
    required_evidence: List[Any] = []
    forbidden_assumptions: List[Any] = []
    intent_encoding_guidance: str = ""


class MainControlDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supi: str = ""
    next_agent: Literal["intent_encoding", "optimization_strategy"]
    requested_domains: List[ControlDomain] = Field(default_factory=list)
    serving_slice_change_requested: bool = False
    domain_evidence: Dict[str, List[str]] = Field(default_factory=dict)
    control_semantics: MainControlSemantics = Field(default_factory=MainControlSemantics)
    objective_profile: ObjectiveProfile = Field(default_factory=ObjectiveProfile)
    investigation_targets: List[MainInvestigationTarget] = Field(default_factory=list)
    uncertainty_flags: List[MainUncertaintyFlag] = Field(default_factory=list)
    retry_scope: Optional[MainRetryScope] = Field(default=None)
    required_evidence: List[str] = Field(default_factory=list)
    forbidden_assumptions: List[str] = Field(default_factory=list)
    intent_encoding_guidance: str = ""


class OpenQuestion(BaseModel):
    owner_agent: str = Field(default="")
    question: str = Field(default="")
    blocking: bool = False
    related_domains: List[str] = Field(default_factory=list)


class GlobalControlIntent(BaseModel):
    session_id: str = ""
    snapshot_id: str = ""
    raw_input: str = ""
    supi: str = ""
    round_strategy: MainRoundStrategy = Field(default=MainRoundStrategy.INITIAL_GROUNDING)
    next_agent: Literal["intent_encoding", "optimization_strategy"]
    requested_domains: List[ControlDomain] = Field(default_factory=list)
    serving_slice_change_requested: bool = False
    domain_evidence: Dict[str, List[str]] = Field(default_factory=dict)
    control_semantics: MainControlSemantics = Field(default_factory=MainControlSemantics)
    objective_profile: ObjectiveProfile = Field(default_factory=ObjectiveProfile)
    investigation_targets: List[MainInvestigationTarget] = Field(default_factory=list)
    uncertainty_flags: List[MainUncertaintyFlag] = Field(default_factory=list)
    retry_scope: Optional[MainRetryScope] = Field(default=None)
    required_evidence: List[str] = Field(default_factory=list)
    forbidden_assumptions: List[str] = Field(default_factory=list)
    intent_encoding_guidance: str = ""
    routing_decision: str = Field(default="")
    routing_rationale: str = Field(default="")
    reuse_contract: ReuseContract = Field(default_factory=ReuseContract)


__all__ = [
    "ControlDomain",
    "ControlSemanticMode",
    "GlobalControlIntent",
    "MainControlDecision",
    "MainControlDecisionWire",
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
