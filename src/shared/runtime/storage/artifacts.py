from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from shared.contracts import ArtifactEnvelope

from shared.runtime.core.workspace import AgentWorkspace
from shared.runtime.io.files import ensure_directory, read_json_file, write_json_file_atomic
from shared.runtime.io.paths import runtime_interfaces_root

from .persistence import RuntimePersistence


class ArtifactStore:
    def __init__(self, *, persistence: RuntimePersistence, root: Optional[Path] = None) -> None:
        self.persistence = persistence
        self.root = Path(root) if root is not None else runtime_interfaces_root()
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
        stored_path = self._store_envelope(path, envelope, kind="request")
        enqueued = self.persistence.enqueue_task(
            artifact_id=envelope.artifact_id,
            artifact_type=envelope.artifact_type,
            source_agent=envelope.source_agent,
            target_agent=envelope.target_agent,
            session_id=envelope.scope.session_id,
            snapshot_id=envelope.scope.snapshot_id or "",
            correlation_id=envelope.correlation_id,
        )
        if not enqueued:
            raise RuntimeError(f"Failed to enqueue request artifact {envelope.artifact_id}.")
        return stored_path

    def write_response(self, envelope: ArtifactEnvelope) -> Path:
        path = self.response_dir(envelope.source_agent, envelope.target_agent) / f"{envelope.artifact_id}.json"
        return self._store_envelope(path, envelope, kind="response")

    def read_artifact(self, path: Path) -> ArtifactEnvelope:
        payload = read_json_file(path)
        return ArtifactEnvelope.model_validate(payload)

    def list_artifacts(self, source_agent: str, target_agent: str, *, kind: str) -> Iterable[Path]:
        if kind == "request":
            directory = self.request_dir(source_agent, target_agent)
        elif kind == "response":
            directory = self.response_dir(source_agent, target_agent)
        else:
            raise ValueError(f"Unsupported artifact kind: {kind}")
        return sorted(directory.glob("*.json"))

    def _store_envelope(self, path: Path, envelope: ArtifactEnvelope, *, kind: str) -> Path:
        stored_path = write_json_file_atomic(path, envelope.model_dump(mode="json"))
        recorded = self.persistence.record_artifact(
            artifact_id=envelope.artifact_id,
            correlation_id=envelope.correlation_id,
            session_id=envelope.scope.session_id,
            snapshot_id=envelope.scope.snapshot_id or "",
            source_agent=envelope.source_agent,
            target_agent=envelope.target_agent,
            artifact_type=envelope.artifact_type,
            kind=kind,
            path=str(stored_path),
            payload=envelope.payload,
        )
        if not recorded:
            raise RuntimeError(f"Failed to record {kind} artifact metadata for {envelope.artifact_id}.")
        return stored_path


class ArtifactCache:
    def __init__(self, workspace: AgentWorkspace) -> None:
        self.workspace = workspace
        self.received_dir = workspace.cache_dir / "received"
        self.produced_dir = workspace.cache_dir / "produced"
        self.state_dir = workspace.cache_dir / "state"
        ensure_directory(self.received_dir)
        ensure_directory(self.produced_dir)
        ensure_directory(self.state_dir)

    def cache_received(self, envelope: ArtifactEnvelope) -> Path:
        path = self.received_dir / f"{envelope.artifact_id}.json"
        return write_json_file_atomic(path, envelope.model_dump(mode="json"))

    def cache_produced(self, envelope: ArtifactEnvelope) -> Path:
        path = self.produced_dir / f"{envelope.artifact_id}.json"
        return write_json_file_atomic(path, envelope.model_dump(mode="json"))

    def write_state(self, name: str, payload: Dict[str, Any]) -> Path:
        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise ValueError("state name is required")
        path = self.state_dir / f"{normalized_name}.json"
        return write_json_file_atomic(path, payload)

    def read_state(self, name: str) -> Dict[str, Any]:
        path = self.state_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        return read_json_file(path)
