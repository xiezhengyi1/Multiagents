from typing import Any, Dict

from pydantic import BaseModel, Field


class FeedbackReport(BaseModel):
    execution_status: str = Field(description="Deterministic execution result: Success, Partial Success, or Failed")
    performance_metrics: str = Field(description="Serialized dispatch receipts and assurance verdicts produced by the execution controller")
    violation_details: str = Field(description="Concrete deterministic failure or violation details, or None")
    failure_scope: str = Field(default="none", description="Raw failure scope observed during execution: qos, mobility, mixed, compile, or none")
    feedback_payload: Dict[str, Any] = Field(default_factory=dict, description="Machine-readable execution facts for upstream agents")
    dispatch_attempts: int = Field(default=0, description="How many dispatch attempts the execution controller used in this round")
    committed_snapshot_id: str = Field(default="", description="Snapshot id produced by the final commit writeback")

__all__ = ["FeedbackReport"]
