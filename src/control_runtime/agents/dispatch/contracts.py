from typing import Any, Dict

from pydantic import BaseModel, Field


class FeedbackReport(BaseModel):
    execution_status: str = Field(description="Deterministic execution result: Success, Partial Success, or Failed")
    performance_metrics: str = Field(description="Serialized dispatch receipts and assurance verdicts produced by the execution controller")
    violation_details: str = Field(description="Concrete deterministic failure or violation details, or None")
    correction_suggestion: str = Field(description="Actionable remediation guidance derived from execution/controller rules")
    recommended_consumer: str = Field(default="none", description="Next component that should consume feedback: intent_encoding, optimization_strategy, or none")
    recommended_action: str = Field(default="none", description="Execution-controller follow-up action: commit, feedback, or none")
    failure_scope: str = Field(default="none", description="Failure scope classified by the execution controller: qos, mobility, mixed, compile, or none")
    feedback_payload: Dict[str, Any] = Field(default_factory=dict, description="Machine-readable deterministic payload for the next consumer")
    dispatch_attempts: int = Field(default=0, description="How many dispatch attempts the execution controller used in this round")
    committed_snapshot_id: str = Field(default="", description="Snapshot id produced by the final commit writeback")

__all__ = ["FeedbackReport"]
