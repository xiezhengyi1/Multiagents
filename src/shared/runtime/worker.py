from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from .api import AgentWorkspace, ArtifactCache, ArtifactEnvelope, ArtifactStore, FileTaskQueue


class ArtifactWorkerMixin:
    agent_name: str
    workspace: AgentWorkspace
    cache: ArtifactCache
    artifact_store: ArtifactStore
    queue: FileTaskQueue

    def init_worker_runtime(self, *, lease_seconds: int = 60) -> None:
        self.workspace = AgentWorkspace.for_agent(self.agent_name)
        self.cache = ArtifactCache(self.workspace)
        self.artifact_store = ArtifactStore()
        self.queue = FileTaskQueue(self.agent_name, lease_seconds=lease_seconds)

    def ensure_worker_runtime_initialized(self) -> None:
        agent_name = str(getattr(self, "agent_name", "") or "").strip()
        if not agent_name:
            raise RuntimeError("agent_name must be defined before worker runtime initialization")
        if not hasattr(self, "workspace") or not hasattr(self, "cache") or not hasattr(self, "artifact_store") or not hasattr(self, "queue"):
            self.init_worker_runtime()

    def expected_request_type(self) -> str:
        raise NotImplementedError

    def response_artifact_type(self) -> str:
        raise NotImplementedError

    def handle_artifact(self, envelope: ArtifactEnvelope) -> Any:
        raise NotImplementedError

    def build_response_payload(self, result: Any) -> dict:
        if hasattr(result, "model_dump"):
            dumped = result.model_dump(mode="json")
            if isinstance(dumped, dict):
                return dumped
        if isinstance(result, dict):
            return result
        raise TypeError(f"Unsupported worker result payload: {type(result).__name__}")

    def consume_request_artifact(self, request_path: Path) -> Path:
        self.ensure_worker_runtime_initialized()
        envelope = self.artifact_store.read_artifact(request_path)
        if envelope.target_agent != self.agent_name:
            raise ValueError(f"artifact target_agent mismatch: expected {self.agent_name}, got {envelope.target_agent}")
        if envelope.artifact_type != self.expected_request_type():
            raise ValueError(f"artifact type mismatch: expected {self.expected_request_type()}, got {envelope.artifact_type}")
        if not self.queue.claim_artifact(envelope):
            raise RuntimeError(f"artifact {envelope.artifact_id} is already claimed for {self.agent_name}")

        try:
            self.cache.cache_received(envelope)
            result = self.handle_artifact(envelope)
            response_envelope = ArtifactEnvelope(
                artifact_type=self.response_artifact_type(),
                source_agent=self.agent_name,
                target_agent=envelope.source_agent,
                session_id=envelope.session_id,
                snapshot_id=envelope.snapshot_id,
                correlation_id=envelope.correlation_id,
                upstream_artifact_ids=[envelope.artifact_id],
                payload=self.build_response_payload(result),
            )
            response_path = self.artifact_store.write_response(response_envelope)
            self.cache.cache_produced(response_envelope)
            self.queue.mark_success(envelope)
            return response_path
        except Exception as exc:
            self.queue.mark_failure(envelope, exc)
            raise

    def poll_once(self) -> Optional[Path]:
        for request_path in self.queue.list_pending_requests():
            envelope = self.artifact_store.read_artifact(request_path)
            if envelope.target_agent != self.agent_name:
                continue
            try:
                return self.consume_request_artifact(request_path)
            except RuntimeError as exc:
                if "already claimed" in str(exc):
                    continue
                raise
        return None

    def run_forever(self, *, poll_interval: float = 0.5) -> None:
        while True:
            processed = self.poll_once()
            if processed is None:
                time.sleep(max(0.05, float(poll_interval)))

    def write_request_artifact(self, envelope: ArtifactEnvelope) -> Path:
        self.ensure_worker_runtime_initialized()
        return self.artifact_store.write_request(envelope)
