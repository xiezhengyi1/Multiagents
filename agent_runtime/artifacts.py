from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .workspace import AgentWorkspace, runtime_root


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(default_factory=lambda: f"artifact-{uuid4()}")
    artifact_type: str
    schema_version: str = Field(default="1.0")
    source_agent: str
    target_agent: str
    session_id: str = Field(default="")
    snapshot_id: str = Field(default="")
    created_at: str = Field(default_factory=_utc_now_iso)
    correlation_id: str = Field(default_factory=lambda: f"corr-{uuid4()}")
    payload: Dict[str, Any] = Field(default_factory=dict)
    upstream_artifact_ids: list[str] = Field(default_factory=list)


class ArtifactStore:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else runtime_root() / "interfaces"
        self.root.mkdir(parents=True, exist_ok=True)

    def request_dir(self, source_agent: str, target_agent: str) -> Path:
        path = self.root / f"{source_agent}__{target_agent}" / "requests"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def response_dir(self, source_agent: str, target_agent: str) -> Path:
        path = self.root / f"{source_agent}__{target_agent}" / "responses"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_request(self, envelope: ArtifactEnvelope) -> Path:
        path = self.request_dir(envelope.source_agent, envelope.target_agent) / f"{envelope.artifact_id}.json"
        self._write_atomic(path, envelope.model_dump(mode="json"))
        return path

    def write_response(self, envelope: ArtifactEnvelope) -> Path:
        path = self.response_dir(envelope.source_agent, envelope.target_agent) / f"{envelope.artifact_id}.json"
        self._write_atomic(path, envelope.model_dump(mode="json"))
        return path

    def read_artifact(self, path: Path) -> ArtifactEnvelope:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return ArtifactEnvelope.model_validate(payload)

    def list_artifacts(self, source_agent: str, target_agent: str, *, kind: str) -> Iterable[Path]:
        if kind == "request":
            directory = self.request_dir(source_agent, target_agent)
        elif kind == "response":
            directory = self.response_dir(source_agent, target_agent)
        else:
            raise ValueError(f"Unsupported artifact kind: {kind}")
        return sorted(directory.glob("*.json"))

    @staticmethod
    def _write_atomic(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp-{uuid4().hex}")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)


class ArtifactCache:
    def __init__(self, workspace: AgentWorkspace) -> None:
        self.workspace = workspace
        self.received_dir = workspace.cache_dir / "received"
        self.produced_dir = workspace.cache_dir / "produced"
        self.state_dir = workspace.cache_dir / "state"
        self.received_dir.mkdir(parents=True, exist_ok=True)
        self.produced_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def cache_received(self, envelope: ArtifactEnvelope) -> Path:
        path = self.received_dir / f"{envelope.artifact_id}.json"
        self._write_atomic(path, envelope.model_dump(mode="json"))
        return path

    def cache_produced(self, envelope: ArtifactEnvelope) -> Path:
        path = self.produced_dir / f"{envelope.artifact_id}.json"
        self._write_atomic(path, envelope.model_dump(mode="json"))
        return path

    def write_state(self, name: str, payload: Dict[str, Any]) -> Path:
        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise ValueError("state name is required")
        path = self.state_dir / f"{normalized_name}.json"
        self._write_atomic(path, payload)
        return path

    def read_state(self, name: str) -> Dict[str, Any]:
        path = self.state_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_atomic(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp-{uuid4().hex}")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
