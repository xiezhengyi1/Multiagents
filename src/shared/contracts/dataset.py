from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from .common import ContractModel, NonEmptyStr


ManifestStatus = Literal["writing", "finalized"]


class TraceSegmentManifest(ContractModel):
    dataset_run_id: NonEmptyStr
    worker_id: NonEmptyStr
    agent_name: NonEmptyStr
    relative_path: NonEmptyStr
    status: ManifestStatus
    created_at: datetime
    updated_at: datetime
    record_count: int = Field(ge=0)
    sha256: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_finalized_checksum(self) -> "TraceSegmentManifest":
        if self.status == "finalized" and self.sha256 is None:
            raise ValueError("finalized trace segment requires sha256")
        if self.status == "writing" and self.sha256 is not None:
            raise ValueError("writing trace segment cannot claim a finalized sha256")
        return self


class TraceWorkerManifest(ContractModel):
    dataset_run_id: NonEmptyStr
    worker_id: NonEmptyStr
    status: Literal["finalized"] = "finalized"
    finalized_at: datetime
    record_count: int = Field(ge=0)
    segments: list[TraceSegmentManifest]

    @model_validator(mode="after")
    def validate_segments(self) -> "TraceWorkerManifest":
        if any(segment.status != "finalized" for segment in self.segments):
            raise ValueError("worker manifest may contain only finalized segments")
        if any(segment.dataset_run_id != self.dataset_run_id for segment in self.segments):
            raise ValueError("worker manifest segments must belong to its dataset run")
        if any(segment.worker_id != self.worker_id for segment in self.segments):
            raise ValueError("worker manifest segments must belong to its worker")
        if self.record_count != sum(segment.record_count for segment in self.segments):
            raise ValueError("worker manifest record_count must equal its segment total")
        return self


class TraceDatasetManifest(ContractModel):
    dataset_run_id: NonEmptyStr
    status: Literal["finalized"] = "finalized"
    finalized_at: datetime
    record_count: int = Field(ge=0)
    workers: list[TraceWorkerManifest]

    @model_validator(mode="after")
    def validate_workers(self) -> "TraceDatasetManifest":
        if any(worker.dataset_run_id != self.dataset_run_id for worker in self.workers):
            raise ValueError("dataset manifest workers must belong to its dataset run")
        worker_ids = [worker.worker_id for worker in self.workers]
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError("dataset manifest worker_id values must be unique")
        if self.record_count != sum(worker.record_count for worker in self.workers):
            raise ValueError("dataset manifest record_count must equal its worker total")
        return self


__all__ = [
    "ManifestStatus",
    "TraceDatasetManifest",
    "TraceSegmentManifest",
    "TraceWorkerManifest",
]
