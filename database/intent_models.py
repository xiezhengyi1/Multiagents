from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ManagedIntent(Base):
    __tablename__ = "managed_intent"

    intent_id: Mapped[str] = mapped_column(String, primary_key=True)
    current_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    origin: Mapped[str] = mapped_column(String, nullable=False)
    bindings: Mapped[list[dict[str, object]]] = mapped_column(JSONB, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class ManagedIntentVersion(Base):
    __tablename__ = "managed_intent_version"

    intent_id: Mapped[str] = mapped_column(
        ForeignKey("managed_intent.intent_id"), primary_key=True
    )
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    specification: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class IntentAuditEvent(Base):
    __tablename__ = "intent_audit_event"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    intent_id: Mapped[str] = mapped_column(
        ForeignKey("managed_intent.intent_id"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    previous_status: Mapped[str | None] = mapped_column(String)
    next_status: Mapped[str] = mapped_column(String, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )


__all__ = ["IntentAuditEvent", "ManagedIntent", "ManagedIntentVersion"]
