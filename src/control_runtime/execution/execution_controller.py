from __future__ import annotations

from control_runtime.simulation import ExecutionAction, SimulatorAdapter

from .assurance_evaluator import AssuranceEvaluator
from .contracts import ExecutionOutcome
from .policy_batch_executor import PolicyBatchExecutor


class ExecutionController:
    """Execute a typed action and optionally evaluate explicit objective metrics."""

    def __init__(
        self,
        *,
        backend: SimulatorAdapter,
        assurance_evaluator: AssuranceEvaluator | None = None,
    ) -> None:
        self.executor = PolicyBatchExecutor(backend=backend)
        self.assurance_evaluator = assurance_evaluator

    def execute(self, action: ExecutionAction) -> ExecutionOutcome:
        feedback = self.executor.execute(action)
        assurance = (
            self.assurance_evaluator.evaluate(feedback)
            if self.assurance_evaluator is not None
            else None
        )
        return ExecutionOutcome(
            action=action,
            feedback=feedback,
            assurance=assurance,
        )


__all__ = ["ExecutionController"]
