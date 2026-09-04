from __future__ import annotations

from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class SemanticKnowledge(Base):
    __tablename__ = "semantic_knowledge"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    category: Mapped[str | None] = mapped_column(String, index=True)
    value: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


__all__ = ["SemanticKnowledge"]
