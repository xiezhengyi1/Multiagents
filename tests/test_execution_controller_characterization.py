"""Characterization tests for the execution transaction boundary."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://offline:offline@127.0.0.1:5432/offline_test",
)

import control_runtime.execution.execution_controller as execution_controller
import control_runtime.execution.policy_batch_executor as policy_batch_executor
from control_runtime.execution.assurance_evaluator import AssuranceVerdict
from control_runtime.execution.execution_controller import ExecutionController
from control_runtime.execution.policy_batch_executor import PolicyBatchExecutor
from control_runtime.execution.policy_compiler import PolicyCompiler


def _policy(
    policy_id: str,
    *,
    policy_type: str = "SmPolicyDecision",
    flow_id: str = "flow-1",
) -> dict[str, object]:
    return {
        "policy_id": policy_id,
        "policy_type": policy_type,
        "supi": "imsi-001",
        "flow_id": flow_id,
        "target_type": "flow",
    }


def _strategy(
    policies: list[dict[str, object]],
    *,
    snapshot_id: str = "snapshot-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        session_id="session-1",
        snapshot_id=snapshot_id,
        supi="imsi-001",
        all_policies=policies,
    )


def _controller(**overrides: object) -> ExecutionController:
    defaults: dict[str, object] = {
        "compiler": SimpleNamespace(),
        "dispatch_policy": lambda *_args, **_kwargs: {"status": "success"},
        "assurance_evaluator": SimpleNamespace(),
        "commit_policies": lambda **_kwargs: None,
        "logger": object(),
    }
    defaults.update(overrides)
    return ExecutionController(**defaults)  # type: ignore[arg-type]


def _silence_runtime_logging(monkeypatch) -> None:
    monkeypatch.setattr(execution_controller, "log_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(execution_controller, "log_timing", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(policy_batch_executor, "log_timing", lambda *_args, **_kwargs: None)


def test_compile_policy_absorbs_the_policy_guard_for_unsupported_types() -> None:
    with pytest.raises(ValueError, match="unsupported policy_type"):
        PolicyCompiler.compile_policy(
            {
                "policy_id": "custom-1",
                "policy_type": "UnsupportedPolicy",
                "supi": "imsi-001",
                "app_id": "app-1",
                "flow_id": "flow-1",
                "target_type": "flow",
                "policy_details": {},
            }
        )


def test_execute_success_compiles_dispatches_assures_and_commits_without_snapshot_contract(monkeypatch) -> None:
    _silence_runtime_logging(monkeypatch)
    calls: list[str] = []
    policy = _policy("sm-1")

    class Compiler:
        def compile_plan(self, strategy_output):
            calls.append("compile")
            return SimpleNamespace(
                session_id=strategy_output.session_id,
                snapshot_id=strategy_output.snapshot_id,
                supi=strategy_output.supi,
                policies=list(strategy_output.all_policies),
            )

    def dispatch_policy(*_args, **_kwargs):
        calls.append("dispatch")
        return {"status": "success"}

    def commit_policies(**kwargs):
        calls.append("commit")
        assert kwargs == {
            "supi": "imsi-001",
            "policies": [policy],
            "session_id": "session-1",
            "snapshot_id": "snapshot-1",
        }

    controller = _controller(
        compiler=Compiler(),
        dispatch_policy=dispatch_policy,
        commit_policies=commit_policies,
    )

    def post_apply_assurance(**_kwargs):
        calls.append("assurance")
        return AssuranceVerdict(policy_id="sm-1", flow_id="flow-1", status="satisfied", reason="ok")

    monkeypatch.setattr(controller.policy_batch_executor, "_post_apply_assurance_verdict", post_apply_assurance)

    outcome = controller.execute(_strategy([policy]))

    assert outcome.execution_status == "Success"
    assert outcome.dispatch_attempts == 1
    assert calls == ["compile", "dispatch", "assurance", "commit"]
    assert "committed_snapshot_id" not in outcome.to_dict()
    assert "committed_snapshot_id" not in json.loads(outcome.performance_metrics)


def test_execute_partial_dispatch_failure_preserves_scope_attempts_and_retry_forbidden(monkeypatch) -> None:
    _silence_runtime_logging(monkeypatch)
    qos_policy = _policy("sm-1")
    mobility_policy = _policy(
        "am-1",
        policy_type="PcfAmPolicyControlPolicyAssociation",
        flow_id="flow-2",
    )
    commits: list[dict[str, object]] = []

    class Compiler:
        def compile_plan(self, strategy_output):
            return SimpleNamespace(
                session_id=strategy_output.session_id,
                snapshot_id=strategy_output.snapshot_id,
                supi=strategy_output.supi,
                policies=list(strategy_output.all_policies),
            )

    def dispatch_policy(_policy_type, candidate, **_kwargs):
        if candidate["policy_id"] == "sm-1":
            return {"status": "success"}
        return {"status": "pending", "operation_id": "operation-2"}

    controller = _controller(
        compiler=Compiler(),
        dispatch_policy=dispatch_policy,
        commit_policies=lambda **kwargs: commits.append(kwargs),
    )
    monkeypatch.setattr(
        controller.policy_batch_executor,
        "_post_apply_assurance_verdict",
        lambda **_kwargs: AssuranceVerdict(policy_id="sm-1", flow_id="flow-1", status="satisfied", reason="ok"),
    )

    outcome = controller.execute(_strategy([qos_policy, mobility_policy]))
    metrics = json.loads(outcome.performance_metrics)

    assert outcome.execution_status == "Partial Success"
    assert outcome.failure_scope == "mobility"
    assert outcome.dispatch_attempts == 2
    assert outcome.feedback_payload["phase"] == "execution"
    assert outcome.feedback_payload["successful_policy_ids"] == ["sm-1"]
    assert outcome.feedback_payload["failed_policy_ids"] == ["am-1"]
    assert outcome.feedback_payload["retry_forbidden"] is True
    assert metrics["dispatch_attempts"] == 2
    assert metrics["dispatch_results"][1]["status"] == "failed"
    assert metrics["dispatch_results"][1]["feedback_payload"]["retry_forbidden"] is True
    assert commits == []


def test_execute_assurance_failure_does_not_commit(monkeypatch) -> None:
    _silence_runtime_logging(monkeypatch)
    policy = _policy("sm-1")
    commits: list[dict[str, object]] = []

    class Compiler:
        def compile_plan(self, strategy_output):
            return SimpleNamespace(
                session_id=strategy_output.session_id,
                snapshot_id=strategy_output.snapshot_id,
                supi=strategy_output.supi,
                policies=list(strategy_output.all_policies),
            )

    controller = _controller(
        compiler=Compiler(),
        dispatch_policy=lambda *_args, **_kwargs: {"status": "success"},
        commit_policies=lambda **kwargs: commits.append(kwargs),
    )
    monkeypatch.setattr(
        controller.policy_batch_executor,
        "_post_apply_assurance_verdict",
        lambda **_kwargs: AssuranceVerdict(
            policy_id="sm-1",
            flow_id="flow-1",
            status="violated",
            reason="latency SLA violated",
        ),
    )

    outcome = controller.execute(_strategy([policy]))
    metrics = json.loads(outcome.performance_metrics)

    assert outcome.execution_status == "Failed"
    assert outcome.failure_scope == "qos"
    assert outcome.dispatch_attempts == 1
    assert commits == []
    assert metrics["failure_summaries"][0]["phase"] == "assurance"
    assert metrics["assurance_results"][0]["status"] == "violated"


def test_policy_batch_executor_owns_explicit_dispatch_and_assurance_dependencies(monkeypatch) -> None:
    _silence_runtime_logging(monkeypatch)
    calls: list[str] = []

    def dispatch_policy(*_args, **_kwargs):
        calls.append("dispatch")
        return {"status": "success"}

    class AssuranceEvaluator:
        def evaluate(self, **_kwargs):
            calls.append("assurance")
            return AssuranceVerdict(policy_id="sm-1", flow_id="flow-1", status="satisfied", reason="ok")

    evaluator = AssuranceEvaluator()
    logger = object()
    controller = _controller(
        dispatch_policy=dispatch_policy,
        assurance_evaluator=evaluator,
        logger=logger,
    )
    executor = controller.policy_batch_executor

    assert isinstance(executor, PolicyBatchExecutor)
    assert executor.dispatch_policy is dispatch_policy
    assert executor.assurance_evaluator is evaluator
    assert executor.logger is logger
    assert not hasattr(controller, "_run_policies")
    assert not hasattr(controller, "_dispatch_single_policy")

    monkeypatch.setattr(executor, "_post_apply_assurance_verdict", lambda **_kwargs: None)
    batch = executor.run(
        policies=[_policy("sm-1")],
        session_id="session-1",
        snapshot_id="snapshot-1",
    )

    assert calls == ["dispatch", "assurance"]
    assert batch.total_dispatch_attempts == 1
    assert batch.execution_receipts[0]["status"] == "success"
    assert batch.assurance_results[0]["status"] == "satisfied"
    assert batch.failures == []
