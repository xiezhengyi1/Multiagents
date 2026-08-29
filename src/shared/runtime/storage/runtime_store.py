from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Dict, Iterator

from .persistence import RuntimePersistence


logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _summary(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return {"keys": sorted(payload.keys())[:20]}
    if hasattr(payload, "model_dump"):
        dumped = payload.model_dump(mode="json")
        if isinstance(dumped, dict):
            return _summary(dumped)
    return {"type": type(payload).__name__}


class SqlAlchemyRuntimePersistence(RuntimePersistence):
    """Runtime persistence backed by application-provided SQLAlchemy objects."""

    def __init__(self, *, session_factory: Callable[[], Any], orm_module: Any) -> None:
        self.session_factory = session_factory
        self.orm = orm_module

    @contextmanager
    def session_scope(self) -> Iterator[Any]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

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
    ) -> bool:
        if not artifact_id:
            return False
        try:
            with self.session_scope() as session:
                row = session.query(self.orm.AgentArtifact).filter(
                    self.orm.AgentArtifact.artifact_id == artifact_id
                ).first()
                if row is None:
                    row = self.orm.AgentArtifact(artifact_id=artifact_id)
                    session.add(row)
                row.correlation_id = correlation_id or None
                row.session_id = session_id or None
                row.snapshot_id = snapshot_id or None
                row.source_agent = source_agent
                row.target_agent = target_agent
                row.artifact_type = artifact_type
                row.kind = kind
                row.path = path
                row.payload_summary = _summary(payload)
            return True
        except Exception as exc:
            logger.warning("Failed to record artifact %s: %s", artifact_id, exc)
            return False

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
    ) -> bool:
        if not artifact_id or not target_agent:
            return False
        try:
            with self.session_scope() as session:
                row = session.query(self.orm.AgentTask).filter(
                    self.orm.AgentTask.artifact_id == artifact_id,
                    self.orm.AgentTask.target_agent == target_agent,
                ).first()
                if row is None:
                    row = self.orm.AgentTask(
                        artifact_id=artifact_id,
                        artifact_type=artifact_type,
                        source_agent=source_agent,
                        target_agent=target_agent,
                        session_id=session_id or None,
                        snapshot_id=snapshot_id or None,
                        correlation_id=correlation_id or None,
                        max_attempts=max(1, int(max_attempts)),
                    )
                    session.add(row)
                row.status = "queued"
                row.last_error = None
            return True
        except Exception as exc:
            logger.warning("Failed to enqueue task for artifact %s: %s", artifact_id, exc)
            return False

    def acquire_task_lease(
        self,
        *,
        artifact_id: str,
        target_agent: str,
        lease_owner: str,
        lease_seconds: int = 60,
    ) -> bool:
        if not artifact_id or not target_agent or not lease_owner:
            return False
        now = _utcnow()
        try:
            with self.session_scope() as session:
                row = session.query(self.orm.AgentTask).filter(
                    self.orm.AgentTask.artifact_id == artifact_id,
                    self.orm.AgentTask.target_agent == target_agent,
                ).first()
                if row is None:
                    return False
                if row.status in {"succeeded", "dead_letter"}:
                    return False
                if row.lease_expires_at and row.lease_expires_at > now and row.lease_owner and row.lease_owner != lease_owner:
                    return False
                row.status = "running"
                row.lease_owner = lease_owner
                row.lease_expires_at = now + timedelta(seconds=max(1, int(lease_seconds)))
                row.attempts = int(row.attempts or 0) + 1
            return True
        except Exception as exc:
            logger.warning("Failed to lease task %s/%s: %s", target_agent, artifact_id, exc)
            return False

    def complete_task(
        self,
        *,
        artifact_id: str,
        target_agent: str,
        status: str,
        error: str = "",
    ) -> bool:
        try:
            with self.session_scope() as session:
                row = session.query(self.orm.AgentTask).filter(
                    self.orm.AgentTask.artifact_id == artifact_id,
                    self.orm.AgentTask.target_agent == target_agent,
                ).first()
                if row is None:
                    return False
                normalized = str(status or "").strip().lower()
                if normalized == "failed" and int(row.attempts or 0) >= int(row.max_attempts or 3):
                    row.status = "dead_letter"
                else:
                    row.status = normalized
                row.last_error = str(error or "") or None
                row.lease_owner = None
                row.lease_expires_at = None
            return True
        except Exception as exc:
            logger.warning("Failed to complete task %s/%s: %s", target_agent, artifact_id, exc)
            return False

    def get_task_status(self, artifact_id: str, target_agent: str) -> str | None:
        try:
            with self.session_scope() as session:
                row = session.query(self.orm.AgentTask).filter(
                    self.orm.AgentTask.artifact_id == artifact_id,
                    self.orm.AgentTask.target_agent == target_agent,
                ).first()
                return None if row is None else str(row.status or "")
        except Exception as exc:
            logger.warning("Failed to get task status %s/%s: %s", target_agent, artifact_id, exc)
            return None


__all__ = ["SqlAlchemyRuntimePersistence"]
