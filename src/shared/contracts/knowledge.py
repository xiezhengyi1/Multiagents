from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import ContractModel, ExecutionScope, NonEmptyStr


KnowledgeDomain = Literal["sm_policy", "am_policy", "ursp", "cross_domain"]


class KnowledgeQuery(ContractModel):
    scope: ExecutionScope
    query: NonEmptyStr
    domain: KnowledgeDomain
    limit: int = Field(ge=1, le=30)


class KnowledgeCitation(ContractModel):
    result_key: NonEmptyStr
    spec_id: NonEmptyStr
    release: NonEmptyStr
    version: NonEmptyStr
    anchor: NonEmptyStr
    source_url: NonEmptyStr | None = None


class KnowledgeHit(ContractModel):
    citation: KnowledgeCitation
    title: NonEmptyStr
    document_type: NonEmptyStr
    policy_domain: NonEmptyStr
    retrieval_method: NonEmptyStr
    score: float
    content: NonEmptyStr
    object_tags: list[NonEmptyStr] = Field(default_factory=list)


class KnowledgeResponse(ContractModel):
    query: KnowledgeQuery
    hits: list[KnowledgeHit]


__all__ = [
    "KnowledgeCitation",
    "KnowledgeDomain",
    "KnowledgeHit",
    "KnowledgeQuery",
    "KnowledgeResponse",
]
