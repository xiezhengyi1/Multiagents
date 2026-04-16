from typing import Any, Dict

from pydantic import BaseModel, Field


class FeedbackSummaryDraft(BaseModel):
    violation_details: str = Field(description="User-facing explanation of the concrete failure or violated assurance target")
    correction_suggestion: str = Field(description="Actionable next step aligned with the deterministic routing decision")


class FeedbackReport(BaseModel):
    execution_status: str = Field(description="Overall execution status: Success, Partial Success, or Failed")
    performance_metrics: str = Field(description="Summary of execution receipts and SLA results")
    violation_details: str = Field(description="Explicit failure or violation details, or None")
    correction_suggestion: str = Field(description="Actionable remediation guidance")
    recommended_consumer: str = Field(default="none", description="Suggested next consumer: intent_encoding, optimization_strategy, or none")
    recommended_action: str = Field(default="none", description="Suggested PDA follow-up action: commit, feedback, or none")
    failure_scope: str = Field(default="none", description="qos, mobility, mixed, compile, or none")
    feedback_payload: Dict[str, Any] = Field(default_factory=dict, description="Machine-readable payload for the next consumer")
    dispatch_attempts: int = Field(default=0, description="How many dispatch attempts PDA used in this execution round")

__all__ = ["FeedbackReport", "FeedbackSummaryDraft"]
