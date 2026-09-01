from pydantic import model_validator

from control_runtime.execution.contracts import ExecutionResultContract, IntentStatus


class FeedbackReport(ExecutionResultContract):
    intent_status: IntentStatus = "satisfied"

    @model_validator(mode="after")
    def validate_intent_feedback(self) -> "FeedbackReport":
        if self.intent_status == "partially_satisfied":
            if self.feedback_payload.get("intent_status") != "partially_satisfied":
                raise ValueError("partially_satisfied report requires matching feedback_payload intent_status")
        return self

__all__ = ["FeedbackReport"]
