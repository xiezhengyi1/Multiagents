from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from database.connection import create_knowledge_engine, create_session_factory, require_database_url
from database.models import IntentAuditEvent, ManagedIntent, ManagedIntentVersion

from .contracts import IntentBinding, IntentLifecycle, IntentStatus, LifecycleTransition
from .lifecycle import ensure_transition


class PersistentIntentLifecycleManager:
    def __init__(self, database_url: str | None = None) -> None:
        engine = create_knowledge_engine(database_url or require_database_url())
        self._sessions = create_session_factory(engine)

    def create(
        self,
        *,
        intent_id: str,
        bindings: list[IntentBinding],
        origin: str,
        specification: dict[str, Any],
        expires_at: datetime | None = None,
    ) -> IntentLifecycle:
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            record = ManagedIntent(
                intent_id=intent_id,
                current_version=1,
                status="draft",
                origin=origin,
                bindings=[binding.model_dump(mode="json") for binding in bindings],
                expires_at=expires_at,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            session.add(
                ManagedIntentVersion(
                    intent_id=intent_id,
                    version=1,
                    specification=specification,
                    created_at=now,
                )
            )
            session.add(
                IntentAuditEvent(
                    event_id=str(uuid4()),
                    intent_id=intent_id,
                    version=1,
                    event_type="created",
                    previous_status=None,
                    next_status="draft",
                    reason="intent created",
                    created_at=now,
                )
            )
        return self.get(intent_id)

    def revise(
        self, *, intent_id: str, specification: dict[str, Any], reason: str
    ) -> IntentLifecycle:
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            record = session.get(ManagedIntent, intent_id)
            if record is None:
                raise KeyError(intent_id)
            version = record.current_version + 1
            record.current_version = version
            record.updated_at = now
            session.add(
                ManagedIntentVersion(
                    intent_id=intent_id,
                    version=version,
                    specification=specification,
                    created_at=now,
                )
            )
            session.add(
                IntentAuditEvent(
                    event_id=str(uuid4()),
                    intent_id=intent_id,
                    version=version,
                    event_type="revised",
                    previous_status=record.status,
                    next_status=record.status,
                    reason=reason,
                    created_at=now,
                )
            )
        return self.get(intent_id)

    def transition(
        self, *, intent_id: str, next_status: IntentStatus, reason: str
    ) -> LifecycleTransition:
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            record = session.get(ManagedIntent, intent_id)
            if record is None:
                raise KeyError(intent_id)
            ensure_transition(record.status, next_status)
            transition = LifecycleTransition(
                intent_id=intent_id,
                previous_status=record.status,
                next_status=next_status,
                reason=reason,
            )
            record.status = next_status
            record.updated_at = now
            session.add(
                IntentAuditEvent(
                    event_id=str(uuid4()),
                    intent_id=intent_id,
                    version=record.current_version,
                    event_type="transition",
                    previous_status=transition.previous_status,
                    next_status=next_status,
                    reason=reason,
                    created_at=now,
                )
            )
        return transition

    def get(self, intent_id: str) -> IntentLifecycle:
        with self._sessions() as session:
            record = session.get(ManagedIntent, intent_id)
            if record is None:
                raise KeyError(intent_id)
            return IntentLifecycle(
                intent_id=record.intent_id,
                version=record.current_version,
                bindings=[IntentBinding.model_validate(item) for item in record.bindings],
                status=record.status,
                origin=record.origin,
                expires_at=record.expires_at,
            )

    def history(self, intent_id: str) -> list[IntentAuditEvent]:
        with self._sessions() as session:
            return list(
                session.scalars(
                    select(IntentAuditEvent)
                    .where(IntentAuditEvent.intent_id == intent_id)
                    .order_by(IntentAuditEvent.created_at)
                )
            )


__all__ = ["PersistentIntentLifecycleManager"]
