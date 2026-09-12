from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from control_runtime.actors import OutcomeEvaluation, SingleControlActor
from control_runtime.graph import RepairLimits
from control_runtime.workflow import (
    ControlModels,
    ControlTools,
    ControlWorkflow,
    ModelBinding,
    SingleControlTools,
    SingleControlWorkflow,
)
from environment import (
    EnvironmentGenerationRequest,
    EnvironmentModel,
    EnvironmentWorkflow,
)
from knowledge_runtime import KnowledgeQuery, KnowledgeResponse, create_rag_tools


class _Simulator:
    def capture_snapshot(self) -> Any:
        raise AssertionError("workflow construction must not read the simulator")

    def observe(self, request: Any) -> Any:
        raise AssertionError(f"unexpected observation: {request}")

    def execute(self, action: Any) -> Any:
        raise AssertionError(f"unexpected execution: {action}")


class _OutcomeEvaluator:
    def evaluate(self, feedback: Any) -> OutcomeEvaluation:
        raise AssertionError(f"unexpected evaluation: {feedback}")


def _binding(name: str) -> ModelBinding:
    return ModelBinding(model=cast(BaseChatModel, object()), name=name)


def test_workflow_variants_use_separate_dataset_partitions_and_graphs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "control_runtime.actors.model.create_structured_agent",
        lambda **_: object(),
    )
    adapter = _Simulator()
    evaluator = _OutcomeEvaluator()
    multi = ControlWorkflow.create(
        dataset_root=tmp_path,
        episode_id="multi-1",
        simulator_adapter=adapter,
        models=ControlModels(
            main=_binding("main-model"),
            sba=_binding("sba-model"),
            ppa=_binding("ppa-model"),
        ),
        tools=ControlTools(main=(), sba=(), ppa=()),
        repair_limits=RepairLimits(
            local_limits={"main": 2, "sba": 2, "ppa": 2},
            upstream={"main": "episode_controller", "sba": "main", "ppa": "sba"},
            total_limit=6,
            controller_actor="episode_controller",
        ),
        outcome_evaluator=evaluator,
        budget_exhausted_reward_components={"repair_budget": -1.0},
        context_token_budget=100_000,
        count_tokens=len,
        transport_max_attempts=1,
    )
    single = SingleControlWorkflow.create(
        dataset_root=tmp_path,
        episode_id="single-1",
        simulator_adapter=adapter,
        model=_binding("single-model"),
        tools=SingleControlTools(extra=()),
        repair_limits=RepairLimits(
            local_limits={"single_control": 2},
            upstream={"single_control": "episode_controller"},
            total_limit=2,
            controller_actor="episode_controller",
        ),
        outcome_evaluator=evaluator,
        budget_exhausted_reward_components={"repair_budget": -1.0},
        context_token_budget=100_000,
        count_tokens=len,
        transport_max_attempts=1,
    )
    environment = EnvironmentWorkflow.create(
        dataset_root=tmp_path,
        episode_id="environment-1",
        model=EnvironmentModel(
            model=cast(BaseChatModel, object()),
            name="environment-model",
        ),
        repair_limits=RepairLimits(
            local_limits={"environment_generator": 2},
            upstream={"environment_generator": "episode_controller"},
            total_limit=2,
            controller_actor="episode_controller",
        ),
        context_token_budget=100_000,
        count_tokens=len,
        transport_max_attempts=1,
    )

    assert multi.journal.directory == tmp_path / "multi_agent_control" / "multi-1"
    assert single.journal.directory == tmp_path / "single_agent_control" / "single-1"
    assert environment.journal.directory == tmp_path / "environment" / "environment-1"
    assert multi.graph.router.is_allowed(
        source_agent="ppa_validator",
        target_agent="sba",
        message_type="repair_request",
    )
    assert multi.graph.router.is_allowed(
        source_agent="policy_mediator",
        target_agent="execution_dispatcher",
        message_type="diagnosis",
    )
    assert multi.graph.router.is_allowed(
        source_agent="assurance_diagnoser",
        target_agent="episode_controller",
        message_type="diagnosis",
    )
    assert single.graph.router.is_allowed(
        source_agent="episode_controller",
        target_agent="single_control",
        message_type="repair_request",
    )
    assert environment.graph.router.is_allowed(
        source_agent="environment_validator",
        target_agent="environment_generator",
        message_type="repair_request",
    )


def test_single_agent_exposes_the_union_of_sba_and_ppa_observation_tools() -> None:
    tool_names = [tool.name for tool in SingleControlActor.default_tools]

    assert tool_names
    assert len(tool_names) == len(set(tool_names))
    assert "get_sm_ue_flow_catalog" in tool_names
    assert "inspect_mobility_ue_policies" in tool_names


def test_environment_request_has_only_current_generation_fields() -> None:
    request = EnvironmentGenerationRequest(
        scenario_id="S-new",
        objective="stress mobility and policy repair",
        complexity="high",
        target_flow_count=20,
        topology_mode="split",
        stress_mode="mobility",
    )

    assert set(request.model_dump()) == {
        "scenario_id",
        "objective",
        "complexity",
        "target_flow_count",
        "topology_mode",
        "stress_mode",
    }
    with pytest.raises(ValidationError):
        EnvironmentGenerationRequest.model_validate(
            {**request.model_dump(), "schema_version": "v2"}
        )
    with pytest.raises(ValidationError):
        EnvironmentGenerationRequest.model_validate(
            {**request.model_dump(), "output_dir": "legacy"}
        )


class _RagBackend:
    def search(self, *, query: str, category: str | None, limit: int) -> Any:
        return KnowledgeResponse(
            query=KnowledgeQuery(query=query, domain="cross_domain", limit=limit),
            hits=[],
        )

    def get_by_key(self, *, key: str, category: str | None) -> Any:
        return KnowledgeResponse(
            query=KnowledgeQuery(query=key, domain="cross_domain", limit=6),
            hits=[],
        )


def test_rag_is_a_standard_detachable_tool_bundle() -> None:
    tools = create_rag_tools(_RagBackend())

    assert [item.name for item in tools.all] == [
        "search_semantic_knowledge",
        "get_knowledge_by_key",
    ]
    assert "runtime" not in tools.search.args_schema.model_fields
    assert tools.search.invoke({"query": "5QI", "category": "shared", "limit": 3}) == {
        "query": {"query": "5QI", "domain": "cross_domain", "limit": 3},
        "hits": [],
    }


def test_rag_backend_errors_propagate_without_fallback() -> None:
    class BrokenBackend(_RagBackend):
        def search(self, *, query: str, category: str | None, limit: int) -> Any:
            raise RuntimeError("backend unavailable")

    with pytest.raises(RuntimeError, match="backend unavailable"):
        create_rag_tools(BrokenBackend()).search.invoke(
            {"query": "5QI", "category": "shared", "limit": 3}
        )
