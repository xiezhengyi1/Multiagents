from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from shared.contracts import ContractModel, NonEmptyStr

from control_runtime.execution.contracts import (
    CompileFailureMetrics,
    ControllerFeedback,
    DispatchExecutionMetrics,
    ExecutionControllerFeedback,
    ExecutionResultContract,
    IntentStatus,
)


class DispatchFailureFeedback(ContractModel):
    feedback_kind: Literal["dispatch_failure"] = "dispatch_failure"
    supi: NonEmptyStr
    policy_id: NonEmptyStr | None
    flow_id: NonEmptyStr | None
    reason: NonEmptyStr
    dispatch_attempts: int = Field(ge=0)
    target_bindings_at_risk: list[NonEmptyStr]
    policy_objects_at_risk: list[NonEmptyStr]
    reason_by_domain: dict[NonEmptyStr, NonEmptyStr]
    controller_feedback: ControllerFeedback
    failure_scope: Literal["qos", "mobility", "mixed", "compile"]
    failures: list[dict[str, JsonValue]]
    retry_forbidden: bool

    @model_validator(mode="after")
    def validate_controller_scope(self) -> "DispatchFailureFeedback":
        if isinstance(self.controller_feedback, ExecutionControllerFeedback):
            if self.controller_feedback.failure_scope != self.failure_scope:
                raise ValueError("dispatch feedback scope must match controller feedback scope")
            if self.controller_feedback.failures != self.failures:
                raise ValueError("dispatch feedback failures must match controller feedback failures")
            if self.controller_feedback.retry_forbidden != self.retry_forbidden:
                raise ValueError("dispatch feedback retry_forbidden must match controller feedback")
        elif self.failure_scope != "compile":
            raise ValueError("compile controller feedback requires failure_scope=compile")
        return self


class PartialIntentFeedback(ContractModel):
    feedback_kind: Literal["partial_intent"] = "partial_intent"
    intent_status: Literal["partially_satisfied"] = "partially_satisfied"
    intent_blockers: list[NonEmptyStr]
    upstream_requests: list[NonEmptyStr]
    planning_status: Literal["partial_plan"] = "partial_plan"


ReportFeedback = Annotated[
    DispatchFailureFeedback | PartialIntentFeedback,
    Field(discriminator="feedback_kind"),
]


class FeedbackReport(ExecutionResultContract):
    intent_status: IntentStatus = "satisfied"
    feedback_payload: ReportFeedback | None

    @model_validator(mode="after")
    def validate_intent_feedback(self) -> "FeedbackReport":
        if self.intent_status == "partially_satisfied":
            if not isinstance(self.feedback_payload, PartialIntentFeedback):
                raise ValueError("partially_satisfied report requires PartialIntentFeedback")
            if self.execution_status != "Success":
                raise ValueError("partial intent satisfaction requires successful policy execution")
            return self
        if self.execution_status == "Success":
            if self.feedback_payload is not None:
                raise ValueError("satisfied successful report cannot contain feedback_payload")
            return self
        if not isinstance(self.feedback_payload, DispatchFailureFeedback):
            raise ValueError("non-successful report requires DispatchFailureFeedback")
        if self.feedback_payload.failure_scope != self.failure_scope:
            raise ValueError("report feedback failure_scope must match report failure_scope")
        if self.feedback_payload.dispatch_attempts != self.dispatch_attempts:
            raise ValueError("report feedback dispatch_attempts must match report dispatch_attempts")
        if isinstance(self.performance_metrics, DispatchExecutionMetrics):
            if self.feedback_payload.failures != self.performance_metrics.failure_summaries:
                raise ValueError("report feedback failures must match performance metric failure summaries")
        elif isinstance(self.performance_metrics, CompileFailureMetrics):
            if self.feedback_payload.controller_feedback != self.performance_metrics.feedback_payload:
                raise ValueError("report compile feedback must match performance metric feedback")
        return self

__all__ = [
    "DispatchFailureFeedback",
    "FeedbackReport",
    "PartialIntentFeedback",
    "ReportFeedback",
]
