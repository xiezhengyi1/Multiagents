from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from shared.contracts import (
    ArtifactEnvelope,
    ExecutionScope,
    KnowledgeCitation,
    KnowledgeHit,
    KnowledgeQuery,
    KnowledgeResponse,
    MemoryBundle,
    MemoryQuery,
    MemoryRecord,
    validate_scope_lineage,
)


def _scope(
    *,
    session_id: str = "session-1",
    thread_id: str = "thread-1",
    supi: str | None = "imsi-001",
) -> ExecutionScope:
    return ExecutionScope(
        session_id=session_id,
        thread_id=thread_id,
        snapshot_id="snapshot-1",
        supi=supi,
    )


def _grounding_request_payload(scope: ExecutionScope) -> dict[str, object]:
    return {
        "scope": scope.model_dump(mode="json"),
        "user_input": "protect flow",
        "context": "",
        "messages": [],
        "allow_user_interaction": False,
    }


def _policy_plan_payload(scope: ExecutionScope) -> dict[str, object]:
    return {
        "scope": scope.model_dump(mode="json"),
        "planning_status": "needs_upstream_reground",
        "optimizer_result": {
            "objective_breakdown": {},
            "cross_domain_verdicts": [],
        },
        "planning_rationale": {
            "selected_strategy_profile": "await-grounding",
            "decisive_evidence": [],
            "active_constraints": [],
            "explanation": "Grounding evidence is required.",
            "rejected_alternatives": [],
            "main_constraints": [],
            "iea_grounding_basis": [],
            "osa_decision_basis": [],
            "unresolved_gaps": ["grounding evidence"],
        },
        "all_policies": [],
        "partial_policies": [],
        "missing_evidence": ["grounding evidence"],
        "blocked_targets": [],
        "upstream_requests": ["reground policy targets"],
        "planner_conflicts": [],
        "open_questions": [],
    }


def _memory(
    memory_id: str,
    *,
    kind: str,
    scope: ExecutionScope,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        scope=scope,
        kind=kind,
        source_agent="test-agent",
        role="summary" if kind == "summary" else "user",
        content="contract test memory",
        source_trace_ids=["trace-1"],
    )


def _hit(result_key: str) -> KnowledgeHit:
    return KnowledgeHit(
        citation=KnowledgeCitation(
            result_key=result_key,
            spec_id="TS 29.512",
            release="19",
            version="19.1.0",
            anchor="section-1",
        ),
        title="Policy control",
        document_type="clause",
        policy_domain="sm_policy",
        retrieval_method="exact",
        score=1.0,
        content="Standards evidence.",
        object_tags=["policy"],
    )


def test_memory_bundle_enforces_record_kind_and_namespace() -> None:
    query = MemoryQuery(scope=_scope(), text="policy", top_k=3)

    with pytest.raises(ValidationError, match="short_term memory must contain only event"):
        MemoryBundle(
            query=query,
            short_term=[_memory("summary-in-short", kind="summary", scope=query.scope)],
            long_term=[],
        )

    with pytest.raises(ValidationError, match="query thread_id"):
        MemoryBundle(
            query=query,
            short_term=[
                _memory(
                    "wrong-thread",
                    kind="event",
                    scope=_scope(thread_id="thread-2"),
                )
            ],
            long_term=[],
        )

    with pytest.raises(ValidationError, match="query SUPI or session namespace"):
        MemoryBundle(
            query=query,
            short_term=[],
            long_term=[
                _memory(
                    "wrong-subscriber",
                    kind="summary",
                    scope=_scope(session_id="session-2", thread_id="thread-2", supi="imsi-002"),
                )
            ],
        )


def test_memory_contract_rejects_duplicate_provenance_and_coerced_top_k() -> None:
    with pytest.raises(ValidationError, match="source_trace_ids must be unique"):
        MemoryRecord(
            memory_id="event-1",
            scope=_scope(),
            kind="event",
            source_agent="test-agent",
            role="user",
            content="duplicate trace provenance",
            source_trace_ids=["trace-1", "trace-1"],
        )

    with pytest.raises(ValidationError):
        MemoryQuery(scope=_scope(), text="policy", top_k="3")


def test_knowledge_response_enforces_limit_and_unique_citations() -> None:
    query = KnowledgeQuery(
        scope=_scope(),
        query="policy",
        domain="sm_policy",
        limit=1,
    )

    with pytest.raises(ValidationError, match="hit count exceeds query limit"):
        KnowledgeResponse(query=query, hits=[_hit("result-1"), _hit("result-2")])

    duplicate_query = KnowledgeQuery(
        scope=_scope(),
        query="policy",
        domain="sm_policy",
        limit=2,
    )
    with pytest.raises(ValidationError, match="result_key values must be unique"):
        KnowledgeResponse(query=duplicate_query, hits=[_hit("result-1"), _hit("result-1")])

    with pytest.raises(ValidationError):
        KnowledgeQuery(
            scope=_scope(),
            query="policy",
            domain="sm_policy",
            limit="3",
        )


def test_artifact_envelope_has_one_type_discriminator_and_explicit_direction() -> None:
    scope = _scope()
    request = ArtifactEnvelope(
        kind="request",
        artifact_id="artifact-request",
        artifact_type="GroundingDecisionRequest",
        source_agent="main_control",
        target_agent="optimization_strategy",
        scope=scope,
        payload=_grounding_request_payload(scope),
    )
    assert "payload_schema_name" not in request.model_dump(mode="json")

    with pytest.raises(ValidationError, match="request artifact cannot declare upstream"):
        ArtifactEnvelope(
            kind="request",
            artifact_type="GroundingDecisionRequest",
            source_agent="main_control",
            target_agent="optimization_strategy",
            scope=scope,
            payload=_grounding_request_payload(scope),
            upstream_artifact_ids=["artifact-parent"],
        )

    with pytest.raises(ValidationError, match="response artifact must declare"):
        ArtifactEnvelope(
            kind="response",
            artifact_type="PolicyPlanDraft",
            source_agent="optimization_strategy",
            target_agent="main_control",
            scope=scope,
            payload=_policy_plan_payload(scope),
        )

    with pytest.raises(ValidationError, match="must differ"):
        ArtifactEnvelope(
            kind="request",
            artifact_type="GroundingDecisionRequest",
            source_agent="main_control",
            target_agent="main_control",
            scope=scope,
            payload=_grounding_request_payload(scope),
        )


def test_artifact_registry_rejects_unknown_types_directions_and_scope_drift() -> None:
    scope = _scope()
    payload = _grounding_request_payload(scope)

    with pytest.raises(ValidationError, match="unregistered artifact_type"):
        ArtifactEnvelope(
            kind="request",
            artifact_type="UnregisteredArtifact",
            source_agent="main_control",
            target_agent="intent_encoding",
            scope=scope,
            payload=payload,
        )

    with pytest.raises(ValidationError, match="cannot be used as kind=response"):
        ArtifactEnvelope(
            kind="response",
            artifact_type="GroundingDecisionRequest",
            source_agent="intent_encoding",
            target_agent="main_control",
            scope=scope,
            payload=payload,
            upstream_artifact_ids=["artifact-request"],
        )

    drifted_scope = _scope(thread_id="thread-2")
    with pytest.raises(ValidationError, match="payload scope must match"):
        ArtifactEnvelope(
            kind="request",
            artifact_type="GroundingDecisionRequest",
            source_agent="main_control",
            target_agent="intent_encoding",
            scope=scope,
            payload=_grounding_request_payload(drifted_scope),
        )


def test_artifact_response_scope_only_allows_supi_enrichment() -> None:
    upstream = _scope(supi=None)
    validate_scope_lineage(upstream, _scope(supi="imsi-001"))

    with pytest.raises(ValueError, match="scope lineage fields"):
        validate_scope_lineage(upstream, _scope(thread_id="thread-2", supi="imsi-001"))

    with pytest.raises(ValueError, match="established scope.supi"):
        validate_scope_lineage(_scope(supi="imsi-001"), _scope(supi="imsi-002"))


def test_memory_and_artifact_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError):
        MemoryRecord(
            memory_id="event-1",
            scope=_scope(),
            kind="event",
            source_agent="test-agent",
            role="user",
            content="naive timestamp",
            created_at=datetime(2026, 1, 1),
        )

    with pytest.raises(ValidationError):
        scope = _scope()
        ArtifactEnvelope(
            kind="request",
            artifact_type="GroundingDecisionRequest",
            source_agent="main_control",
            target_agent="optimization_strategy",
            scope=scope,
            payload=_grounding_request_payload(scope),
            created_at=datetime(2026, 1, 1),
        )
