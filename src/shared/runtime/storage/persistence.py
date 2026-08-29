from __future__ import annotations

from typing import Any, Protocol


class RuntimePersistence(Protocol):
    """Persistence boundary used by runtime artifact storage and task queues."""

    def record_artifact(
        self,
        *,
        artifact_id: str,
        correlation_id: str,
        session_id: str,
        snapshot_id: str,
        source_agent: str,
        target_agent: str,
        artifact_type: str,
        kind: str,
        path: str,
        payload: Any,
    ) -> bool: ...

    def enqueue_task(
        self,
        *,
        artifact_id: str,
        artifact_type: str,
        source_agent: str,
        target_agent: str,
        session_id: str,
        snapshot_id: str,
        correlation_id: str,
        max_attempts: int = 3,
    ) -> bool: ...

    def acquire_task_lease(
        self,
        *,
        artifact_id: str,
        target_agent: str,
        lease_owner: str,
        lease_seconds: int = 60,
    ) -> bool: ...

    def complete_task(
        self,
        *,
        artifact_id: str,
        target_agent: str,
        status: str,
        error: str = "",
    ) -> bool: ...

    def get_task_status(self, artifact_id: str, target_agent: str) -> str | None: ...


__all__ = ["RuntimePersistence"]
