from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from shared.contracts import (
    ContractModel,
    ExecutionScope,
    NonEmptyStr,
)

from .policy_compiler import CompiledStrategyPlan


ExecutionStatus = Literal["Success", "Partial Success", "Failed"]
IntentStatus = Literal["satisfied", "partially_satisfied", "unsatisfied"]
FailureScope = Literal["none", "qos", "mobility", "mixed", "compile"]


class CompileControllerFeedback(ContractModel):
    phase: Literal["compile"] = "compile"
    error: NonEmptyStr


class ExecutionControllerFeedback(ContractModel):
    phase: Literal["execution"] = "execution"
    failure_scope: Literal["qos", "mobility", "mixed"]
    successful_policy_ids: list[NonEmptyStr]
    failed_policy_ids: list[NonEmptyStr]
    failures: list[dict[str, JsonValue]]
    retry_forbidden: bool


ControllerFeedback = Annotated[
    CompileControllerFeedback | ExecutionControllerFeedback,
    Field(discriminator="phase"),
]


class DomainExecutionReceipts(ContractModel):
    qos: list[dict[str, JsonValue]]
    mobility: list[dict[str, JsonValue]]


class DispatchExecutionMetrics(ContractModel):
    metrics_kind: Literal["dispatch"] = "dispatch"
    policy_plan: CompiledStrategyPlan
    dispatch_results: list[dict[str, JsonValue]]
    assurance_results: list[dict[str, JsonValue]]
    dispatch_attempts: int = Field(ge=0)
    domain_receipts: DomainExecutionReceipts
    failure_summaries: list[dict[str, JsonValue]]

    @model_validator(mode="after")
    def validate_domain_receipts(self) -> "DispatchExecutionMetrics":
        domain_receipt_count = len(self.domain_receipts.qos) + len(self.domain_receipts.mobility)
        if domain_receipt_count != len(self.dispatch_results):
            raise ValueError("domain receipt count must equal dispatch result count")
        return self


class CompileFailureMetrics(ContractModel):
    metrics_kind: Literal["compile_failure"] = "compile_failure"
    dispatch_attempts: Literal[0] = 0
    failure_scope: Literal["compile"] = "compile"
    feedback_payload: CompileControllerFeedback


ExecutionMetrics = Annotated[
    DispatchExecutionMetrics | CompileFailureMetrics,
    Field(discriminator="metrics_kind"),
]


class ExecutionResultContract(ContractModel):
    scope: ExecutionScope
    execution_status: ExecutionStatus
    performance_metrics: ExecutionMetrics
    violation_details: NonEmptyStr
    failure_scope: FailureScope
    dispatch_attempts: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_execution_result(self) -> "ExecutionResultContract":
        if self.dispatch_attempts != self.performance_metrics.dispatch_attempts:
            raise ValueError("dispatch_attempts must match performance_metrics.dispatch_attempts")
        if isinstance(self.performance_metrics, CompileFailureMetrics):
            if self.execution_status != "Failed" or self.failure_scope != "compile":
                raise ValueError("compile_failure metrics require Failed status and compile scope")
            return self
        if self.performance_metrics.policy_plan.scope != self.scope:
            raise ValueError("execution result scope must match performance_metrics.policy_plan.scope")
        if self.execution_status == "Success":
            if self.failure_scope != "none":
                raise ValueError("successful execution requires failure_scope=none")
            if self.performance_metrics.failure_summaries:
                raise ValueError("successful execution cannot contain failure summaries")
        elif self.failure_scope == "none":
            raise ValueError("non-successful execution requires a concrete failure scope")
        return self


class ExecutionOutcome(ExecutionResultContract):
    """Typed controller result before dispatch feedback enrichment."""

    feedback_payload: ControllerFeedback | None

    @model_validator(mode="after")
    def validate_controller_feedback(self) -> "ExecutionOutcome":
        if isinstance(self.performance_metrics, CompileFailureMetrics):
            if not isinstance(self.feedback_payload, CompileControllerFeedback):
                raise ValueError("compile failure requires CompileControllerFeedback")
            if self.feedback_payload != self.performance_metrics.feedback_payload:
                raise ValueError("compile feedback must match performance_metrics feedback")
            return self
        if self.execution_status == "Success":
            if self.feedback_payload is not None:
                raise ValueError("successful controller outcome cannot contain feedback_payload")
            return self
        if not isinstance(self.feedback_payload, ExecutionControllerFeedback):
            raise ValueError("failed dispatch requires ExecutionControllerFeedback")
        if self.feedback_payload.failure_scope != self.failure_scope:
            raise ValueError("controller feedback failure_scope must match outcome failure_scope")
        if self.feedback_payload.failures != self.performance_metrics.failure_summaries:
            raise ValueError("controller feedback failures must match performance metric failure summaries")
        return self


__all__ = [
    "CompileFailureMetrics",
    "CompileControllerFeedback",
    "ControllerFeedback",
    "DispatchExecutionMetrics",
    "DomainExecutionReceipts",
    "ExecutionMetrics",
    "ExecutionControllerFeedback",
    "ExecutionOutcome",
    "ExecutionResultContract",
    "ExecutionStatus",
    "FailureScope",
    "IntentStatus",
]
