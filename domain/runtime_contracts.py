from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ArtifactRef(BaseModel):
    artifact_id: str = Field(default="")
    artifact_type: str = Field(default="")
    source_agent: str = Field(default="")
    target_agent: str = Field(default="")
    session_id: str = Field(default="")
    snapshot_id: str = Field(default="")
    correlation_id: str = Field(default="")
    path: str = Field(default="")


class TaskEnvelope(BaseModel):
    task_id: str = Field(default="")
    artifact: ArtifactRef = Field(default_factory=ArtifactRef)
    status: str = Field(default="queued")
    attempts: int = Field(default=0)
    max_attempts: int = Field(default=3)


class TaskLease(BaseModel):
    artifact_id: str = Field(default="")
    target_agent: str = Field(default="")
    lease_owner: str = Field(default="")
    lease_expires_at: Optional[str] = Field(default=None)


class StageResult(BaseModel):
    stage_name: str = Field(default="")
    status: str = Field(default="")
    artifact_id: str = Field(default="")
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class GraphQuery(BaseModel):
    snapshot_id: str = Field(default="")
    node_type: Optional[str] = Field(default=None)
    edge_type: Optional[str] = Field(default=None)
    owner_key: Optional[str] = Field(default=None)
    metric_name: Optional[str] = Field(default=None)


class GraphDelta(BaseModel):
    snapshot_id: str = Field(default="")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    updated_edges: List[Dict[str, Any]] = Field(default_factory=list)
    updated_metrics: List[Dict[str, Any]] = Field(default_factory=list)
