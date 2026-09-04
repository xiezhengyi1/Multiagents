from __future__ import annotations

from sqlalchemy import text

from database.base import Base
from database.connection import create_knowledge_engine, require_database_url
from database.langchain_pg import ensure_semantic_knowledge_collection
from database.models import SemanticKnowledge


def init_db(database_url: str | None = None) -> None:
    engine = create_knowledge_engine(database_url or require_database_url())
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine, tables=[SemanticKnowledge.__table__])
    ensure_semantic_knowledge_collection()


if __name__ == "__main__":
    init_db()


__all__ = ["init_db"]
