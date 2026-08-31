from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://offline:offline@127.0.0.1:5432/offline_test",
)
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from knowledge_build.scripts import evaluate_pcf_policy_rag
from knowledge_runtime.retrieval.raw import knowledge_tool
from langchain.tools import ToolRuntime
from shared.contracts import ExecutionScope
from shared.runtime import AgentRuntimeContext


def _tool_runtime() -> ToolRuntime:
    return ToolRuntime(
        state={},
        context=AgentRuntimeContext(
            agent_name="test",
            scope=ExecutionScope(session_id="session-1", thread_id="session-1"),
        ),
        config={},
        stream_writer=lambda *_args, **_kwargs: None,
        tool_call_id="call-1",
        store=None,
    )


def _candidate(index: int) -> dict[str, Any]:
    return {
        "id": f"candidate-{index}",
        "metadata": {
            "citation_anchor": f"TS-29.512#{index}",
            "policy_domain": "sm_policy",
        },
    }


def _patch_exact_pipeline(monkeypatch, candidates: list[dict[str, Any]]) -> tuple[list[int], list[str]]:
    pool_limits: list[int] = []
    dense_calls: list[str] = []

    monkeypatch.setattr(knowledge_tool, "_ensure_processed_corpus", lambda: ())
    monkeypatch.setattr(knowledge_tool, "_infer_policy_domain", lambda *_args: "sm_policy")
    monkeypatch.setattr(knowledge_tool, "_query_intent", lambda *_args: "exact_identifier")
    monkeypatch.setattr(knowledge_tool, "_direct_record_matches", lambda *_args: list(candidates))
    monkeypatch.setattr(knowledge_tool, "_glossary_strong_candidates", lambda *_args: [])
    monkeypatch.setattr(knowledge_tool, "_schema_object_graph_candidates", lambda *_args: [])
    monkeypatch.setattr(knowledge_tool, "_schema_definition_candidates", lambda *_args: [])
    monkeypatch.setattr(knowledge_tool, "_sm_qos_requirement_candidates", lambda *_args: [])
    monkeypatch.setattr(knowledge_tool, "_am_rfsp_clause_candidates", lambda *_args: [])
    monkeypatch.setattr(knowledge_tool, "_operation_intent_candidates", lambda *_args: [])
    monkeypatch.setattr(knowledge_tool, "_exact_search_candidates", lambda *_args: [])
    monkeypatch.setattr(knowledge_tool, "_colbert_available", lambda: False)

    def dedupe(items: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
        pool_limits.append(limit)
        return list(items)[:limit]

    def dense(*_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], list[str]]:
        dense_calls.append("called")
        return [], []

    monkeypatch.setattr(knowledge_tool, "_dedupe_candidates", dedupe)
    monkeypatch.setattr(knowledge_tool, "_vector_search_candidates", dense)
    monkeypatch.setattr(
        knowledge_tool,
        "_bge_rerank_candidates",
        lambda _query, _domain, items, *, limit: list(items)[:limit],
    )
    return pool_limits, dense_calls


def test_exact_seed_path_keeps_pool_and_observer_nonintrusive(monkeypatch) -> None:
    candidates = [_candidate(index) for index in range(5)]
    pool_limits, dense_calls = _patch_exact_pipeline(monkeypatch, candidates)

    without_observer = knowledge_tool.run_semantic_knowledge_pipeline("Npcf_SMPolicyControl", limit=5)
    events: list[knowledge_tool.RetrievalStage] = []
    with_observer = knowledge_tool.run_semantic_knowledge_pipeline(
        "Npcf_SMPolicyControl",
        limit=5,
        observer=events.append,
    )

    assert with_observer == without_observer == (candidates, [])
    assert dense_calls == []
    assert pool_limits == [24, 24, 24, 24]
    skipped = {event.name for event in events if event.skipped}
    assert {"expand_cross_spec_candidates", "vector_search_candidates"} <= skipped


def test_pipeline_keeps_raw_evaluator_limit_but_tool_boundary_rejects_it(monkeypatch) -> None:
    captured: dict[str, int] = {}
    monkeypatch.setattr(knowledge_tool, "_ensure_processed_corpus", lambda: ())
    monkeypatch.setattr(knowledge_tool, "_infer_policy_domain", lambda *_args: "sm_policy")

    def fake_pipeline(_query: str, _domain: str, *, limit: int, observer=None):
        captured["raw_limit"] = limit
        return [], []

    monkeypatch.setattr(knowledge_tool, "_run_retrieval_pipeline", fake_pipeline)
    assert knowledge_tool.run_semantic_knowledge_pipeline("raw limit", limit=61) == ([], [])
    assert captured["raw_limit"] == 61

    def fake_public_pipeline(_query: str, *, category=None, limit: int, observer=None):
        captured["tool_limit"] = limit
        return [], []

    monkeypatch.setattr(knowledge_tool, "run_semantic_knowledge_pipeline", fake_public_pipeline)
    with pytest.raises(ValueError, match="less than or equal to 30"):
        knowledge_tool.search_semantic_knowledge.func(
            reason="test limit boundary",
            query="tool limit",
            limit=61,
            runtime=_tool_runtime(),
        )
    assert "tool_limit" not in captured


def test_evaluator_uses_public_pipeline_and_legacy_entrypoints_are_gone(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    candidates = [_candidate(1)]

    def fake_public_pipeline(query: str, *, category=None, limit: int, observer=None):
        captured.update(query=query, category=category, limit=limit)
        return candidates, ["dense warning"]

    monkeypatch.setattr(evaluate_pcf_policy_rag.knowledge_tool, "run_semantic_knowledge_pipeline", fake_public_pipeline)
    result = evaluate_pcf_policy_rag.evaluate_case(
        {
            "id": "case-1",
            "query": "Npcf_SMPolicyControl",
            "category": "sm_policy",
            "policy_domain": "sm_policy",
            "target_kind": "schema",
            "expected_result_keys": ["TS-29.512#1"],
        },
        limit=7,
    )

    assert captured == {"query": "Npcf_SMPolicyControl", "category": "sm_policy", "limit": 7}
    assert result["retrieved_result_keys"] == ["TS-29.512#1"]
    assert result["warnings"] == ["dense warning"]
    assert not hasattr(knowledge_tool, "_assemble_response")
    assert not hasattr(knowledge_tool, "debug_search_semantic_knowledge_pipeline")
    assert not hasattr(evaluate_pcf_policy_rag, "debug_tool_invocation")
