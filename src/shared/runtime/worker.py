from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from shared.contracts import (
    ArtifactType,
    ContractModel,
    ExecutionScope,
    validate_scope_lineage,
)
from .core import AgentWorkspace
from .storage import ArtifactCache, ArtifactEnvelope, ArtifactStore, FileTaskQueue, RuntimePersistence


class ArtifactWorkerMixin:
    agent_name: str
    workspace: AgentWorkspace
    cache: ArtifactCache
    artifact_store: ArtifactStore
    queue: FileTaskQueue
    runtime_persistence: RuntimePersistence

    def init_worker_runtime(self, *, persistence: RuntimePersistence, lease_seconds: int = 60) -> None:
        self.runtime_persistence = persistence
        self.workspace = AgentWorkspace.for_agent(self.agent_name)
        self.cache = ArtifactCache(self.workspace)
        self.artifact_store = ArtifactStore(persistence=persistence)
        self.queue = FileTaskQueue(
            self.agent_name,
            persistence=persistence,
            lease_seconds=lease_seconds,
            artifact_store=self.artifact_store,
        )

    def ensure_worker_runtime_initialized(self) -> None:
        agent_name = str(getattr(self, "agent_name", "") or "").strip()
        if not agent_name:
            raise RuntimeError("agent_name must be defined before worker runtime initialization")
        if not hasattr(self, "workspace") or not hasattr(self, "cache") or not hasattr(self, "artifact_store") or not hasattr(self, "queue"):
            persistence = getattr(self, "runtime_persistence", None)
            if persistence is None:
                raise RuntimeError("worker runtime requires an explicit RuntimePersistence instance")
            self.init_worker_runtime(persistence=persistence)

    def expected_request_type(self) -> ArtifactType:
        raise NotImplementedError

    def response_artifact_type(self) -> ArtifactType:
        raise NotImplementedError

    def handle_artifact(self, envelope: ArtifactEnvelope) -> ContractModel:
        raise NotImplementedError

    @staticmethod
    def build_response_payload(result: ContractModel) -> dict:
        return result.model_dump(mode="json")

    @staticmethod
    def artifact_payload_scope(payload: ContractModel) -> ExecutionScope:
        scope = getattr(payload, "scope")
        if not isinstance(scope, ExecutionScope):
            raise TypeError(f"artifact payload {type(payload).__name__} must expose ExecutionScope")
        return scope

    def cache_received_artifact(
        self,
        *,
        artifact_type: ArtifactType,
        payload: ContractModel,
        source_agent: str = "coordinator",
        scope: ExecutionScope,
        target_agent: Optional[str] = None,
    ) -> ArtifactEnvelope:
        if not hasattr(self, "cache"):
            raise RuntimeError("cache_received_artifact() requires initialized worker runtime")

        payload_scope = self.artifact_payload_scope(payload)
        if payload_scope != scope:
            raise ValueError("received artifact payload scope must match the requested envelope scope")
        envelope = ArtifactEnvelope(
            kind="request",
            artifact_type=artifact_type,
            source_agent=source_agent,
            target_agent=target_agent if target_agent is not None else self.agent_name,
            scope=scope,
            payload=self.build_response_payload(payload),
        )
        self.cache.cache_received(envelope)
        return envelope

    def cache_produced_artifact(
        self,
        *,
        artifact_type: ArtifactType,
        request_envelope: ArtifactEnvelope,
        payload: ContractModel,
        source_agent: Optional[str] = None,
        target_agent: Optional[str] = None,
    ) -> ArtifactEnvelope:
        if not hasattr(self, "cache"):
            raise RuntimeError("cache_produced_artifact() requires initialized worker runtime")

        payload_scope = self.artifact_payload_scope(payload)
        validate_scope_lineage(request_envelope.scope, payload_scope)
        envelope = ArtifactEnvelope(
            kind="response",
            artifact_type=artifact_type,
            source_agent=source_agent if source_agent is not None else self.agent_name,
            target_agent=target_agent if target_agent is not None else request_envelope.source_agent,
            scope=payload_scope,
            correlation_id=request_envelope.correlation_id,
            upstream_artifact_ids=[request_envelope.artifact_id],
            payload=self.build_response_payload(payload),
        )
        self.cache.cache_produced(envelope)
        return envelope

    def consume_request_artifact(self, request_path: Path) -> Path:
        self.ensure_worker_runtime_initialized()
        envelope = self.artifact_store.read_artifact(request_path)
        if envelope.kind != "request":
            raise ValueError(f"artifact kind mismatch: expected request, got {envelope.kind}")
        if envelope.target_agent != self.agent_name:
            raise ValueError(f"artifact target_agent mismatch: expected {self.agent_name}, got {envelope.target_agent}")
        if envelope.artifact_type != self.expected_request_type():
            raise ValueError(f"artifact type mismatch: expected {self.expected_request_type()}, got {envelope.artifact_type}")
        if not self.queue.claim_artifact(envelope):
            raise RuntimeError(f"artifact {envelope.artifact_id} is already claimed for {self.agent_name}")

        try:
            self.cache.cache_received(envelope)
            result = self.handle_artifact(envelope)
            if not isinstance(result, ContractModel):
                raise TypeError("artifact worker must return a ContractModel")
            response_scope = self.artifact_payload_scope(result)
            validate_scope_lineage(envelope.scope, response_scope)
            response_envelope = ArtifactEnvelope(
                kind="response",
                artifact_type=self.response_artifact_type(),
                source_agent=self.agent_name,
                target_agent=envelope.source_agent,
                scope=response_scope,
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
