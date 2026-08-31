from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import Field, JsonValue, model_validator

from .common import CONTRACT_VERSION, ContractModel, ContractVersion, ExecutionScope, NonEmptyStr, utc_now


class ArtifactEnvelope(ContractModel):
    """Versioned model-to-model handoff with explicit scope and provenance."""

    schema_version: ContractVersion = CONTRACT_VERSION
    artifact_id: NonEmptyStr = Field(default_factory=lambda: f"artifact-{uuid4()}")
    artifact_type: NonEmptyStr
    payload_schema_name: NonEmptyStr
    source_agent: NonEmptyStr
    target_agent: NonEmptyStr
    scope: ExecutionScope
    created_at: datetime = Field(default_factory=utc_now)
    correlation_id: NonEmptyStr = Field(default_factory=lambda: f"corr-{uuid4()}")
    payload: dict[str, JsonValue]
    upstream_artifact_ids: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_provenance(self) -> "ArtifactEnvelope":
        if len(self.upstream_artifact_ids) != len(set(self.upstream_artifact_ids)):
            raise ValueError("upstream_artifact_ids must be unique")
        if self.artifact_id in self.upstream_artifact_ids:
            raise ValueError("artifact cannot list itself as upstream")
        return self


__all__ = ["ArtifactEnvelope"]
