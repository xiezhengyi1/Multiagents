from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def require_database_url() -> str:
    value = os.environ.get("DATABASE_URL", "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is required for the PostgreSQL knowledge backend")
    return value


def to_sqlalchemy_url(database_url: str) -> str:
    normalized = database_url.strip()
    if normalized.startswith("postgresql://"):
        return normalized.replace("postgresql://", "postgresql+psycopg://", 1)
    return normalized


def create_knowledge_engine(database_url: str) -> Engine:
    return create_engine(
        to_sqlalchemy_url(database_url),
        pool_pre_ping=True,
        connect_args={"client_encoding": "utf8"},
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


__all__ = [
    "create_knowledge_engine",
    "create_session_factory",
    "require_database_url",
    "to_sqlalchemy_url",
]
