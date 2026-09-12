from __future__ import annotations

from types import SimpleNamespace

from experiments.backends.live_testbed import LiveOutcomeEvaluator


def test_live_outcome_evaluator_reflects_execution_status() -> None:
    evaluator = LiveOutcomeEvaluator()

    success = evaluator.evaluate(SimpleNamespace(status="success"))
    failure = evaluator.evaluate(SimpleNamespace(status="failed"))

    assert success.achieved is True
    assert success.reward_components == {"live_execution": 1.0}
    assert failure.achieved is False
    assert failure.reward_components == {"live_execution": -1.0}
