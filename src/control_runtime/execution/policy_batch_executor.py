from __future__ import annotations

from control_runtime.simulation import (
    ExecutionAction,
    ExecutionFeedback,
    SimulatorAdapter,
)


class PolicyBatchExecutor:
    """Submit one compiled action and enforce receipt identity and ordering."""

    def __init__(self, *, backend: SimulatorAdapter) -> None:
        self.backend = backend

    def execute(self, action: ExecutionAction) -> ExecutionFeedback:
        feedback = self.backend.execute(action)
        operation_ids = [operation.operation_id for operation in action.operations]
        receipt_ids = [receipt.operation_id for receipt in feedback.receipts]
        if receipt_ids != operation_ids:
            raise ValueError(
                "execution receipts must preserve operation order and identity"
            )
        return feedback


__all__ = ["PolicyBatchExecutor"]
