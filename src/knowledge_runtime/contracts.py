from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from shared.contracts import ContractModel, NonEmptyStr

KnowledgeDomain = Literal[
    "sm_policy",
    "am_policy",
    "ursp",
    "cross_domain",
    "mobility",
    "shared",
    "all",
]
CanonicalKnowledgeDomain = Literal["sm_policy", "am_policy", "ursp", "cross_domain"]


class KnowledgeQuery(ContractModel):
    query: NonEmptyStr
    domain: CanonicalKnowledgeDomain
    limit: Annotated[int, Field(strict=True, ge=1, le=30)]


class KnowledgeCitation(ContractModel):
    result_key: NonEmptyStr
    spec_id: NonEmptyStr
    release: NonEmptyStr
    document_version: NonEmptyStr
    anchor: NonEmptyStr
    source_url: NonEmptyStr | None = None


class KnowledgeHit(ContractModel):
    citation: KnowledgeCitation
    title: NonEmptyStr
    document_type: NonEmptyStr
    policy_domain: CanonicalKnowledgeDomain
    retrieval_method: NonEmptyStr
    score: Annotated[float, Field(strict=True, allow_inf_nan=False)]
    content: NonEmptyStr
    object_tags: list[NonEmptyStr] = Field(default_factory=list)

    @field_validator("object_tags")
    @classmethod
    def validate_unique_tags(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("object_tags must be unique")
        return values


class KnowledgeResponse(ContractModel):
    query: KnowledgeQuery
    hits: list[KnowledgeHit]

    @model_validator(mode="after")
    def validate_hits(self) -> KnowledgeResponse:
        if len(self.hits) > self.query.limit:
            raise ValueError("knowledge response hit count exceeds query limit")
        result_keys = [hit.citation.result_key for hit in self.hits]
        if len(result_keys) != len(set(result_keys)):
            raise ValueError("knowledge response result_key values must be unique")
        return self


__all__ = [
    "CanonicalKnowledgeDomain",
    "KnowledgeCitation",
    "KnowledgeDomain",
    "KnowledgeHit",
    "KnowledgeQuery",
    "KnowledgeResponse",
]
