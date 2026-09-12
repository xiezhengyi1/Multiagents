from __future__ import annotations

import pytest

from knowledge_runtime.retrieval.raw import knowledge_tool


def test_pgvector_failure_propagates_without_skipping_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        knowledge_tool,
        "_select_collections",
        lambda query, domain: ["sm-policy"],
    )

    def fail_store(*, collection_name: str) -> None:
        raise RuntimeError(f"{collection_name} unavailable")

    monkeypatch.setattr(knowledge_tool, "get_pgvector_store", fail_store)

    with pytest.raises(RuntimeError, match="sm-policy unavailable"):
        knowledge_tool._vector_search_candidates("5QI", "sm_policy")


def test_selected_colbert_backend_requires_its_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(knowledge_tool, "get_colbert_retriever", lambda: None)

    with pytest.raises(FileNotFoundError, match="ColBERT index not available"):
        knowledge_tool._colbert_search_candidates("RFSP", "am_policy")


def test_get_by_key_does_not_degrade_to_fuzzy_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(knowledge_tool, "_ensure_processed_corpus", lambda: ())
    monkeypatch.setattr(
        knowledge_tool,
        "_infer_policy_domain",
        lambda query, category: "sm_policy",
    )
    monkeypatch.setattr(
        knowledge_tool,
        "_resolve_explicit_key_candidates",
        lambda key, domain: [],
    )

    def fuzzy_search_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("exact key lookup must not call fuzzy search")

    monkeypatch.setattr(
        knowledge_tool,
        "_direct_record_matches",
        fuzzy_search_called,
    )
    monkeypatch.setattr(
        knowledge_tool,
        "_exact_search_candidates",
        fuzzy_search_called,
    )

    response = knowledge_tool.HybridKnowledgeBackend(
        dense_retrieval="colbert"
    ).get_by_key(key="missing-key", category=None)

    assert response.hits == []
