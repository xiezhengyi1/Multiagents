from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from .artifact_registry import ArtifactType, validate_artifact_payload
from .common import ContractModel, ExecutionScope, NonEmptyStr, utc_now


ArtifactKind = Literal["request", "response"]


class ArtifactEnvelope(ContractModel):
    """Model-to-model handoff with explicit scope and provenance."""

    artifact_id: NonEmptyStr = Field(default_factory=lambda: f"artifact-{uuid4()}")
    kind: ArtifactKind
    artifact_type: ArtifactType
    source_agent: NonEmptyStr
    target_agent: NonEmptyStr
    scope: ExecutionScope
    created_at: AwareDatetime = Field(default_factory=utc_now)
    correlation_id: NonEmptyStr = Field(default_factory=lambda: f"corr-{uuid4()}")
    payload: dict[str, JsonValue]
    upstream_artifact_ids: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def validate_registered_payload(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        artifact_type = value.get("artifact_type")
        kind = value.get("kind")
        payload = value.get("payload")
        scope_payload = value.get("scope")
        if not isinstance(artifact_type, str) or kind not in {"request", "response"} or not isinstance(payload, dict):
            return value
        canonical_payload, payload_scope = validate_artifact_payload(artifact_type, kind, payload)
        envelope_scope = ExecutionScope.model_validate(scope_payload)
        if payload_scope != envelope_scope:
            raise ValueError("artifact payload scope must match ArtifactEnvelope scope")
        return {**value, "payload": canonical_payload}

    @model_validator(mode="after")
    def validate_provenance(self) -> "ArtifactEnvelope":
        if self.source_agent == self.target_agent:
            raise ValueError("artifact source_agent and target_agent must differ")
        if len(self.upstream_artifact_ids) != len(set(self.upstream_artifact_ids)):
            raise ValueError("upstream_artifact_ids must be unique")
        if self.artifact_id in self.upstream_artifact_ids:
            raise ValueError("artifact cannot list itself as upstream")
        if self.kind == "request" and self.upstream_artifact_ids:
            raise ValueError("request artifact cannot declare upstream artifacts")
        if self.kind == "response" and not self.upstream_artifact_ids:
            raise ValueError("response artifact must declare at least one upstream artifact")
        return self


__all__ = ["ArtifactEnvelope", "ArtifactKind", "ArtifactType"]
