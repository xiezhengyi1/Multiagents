from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PACKAGE_ROOT / "src"
for candidate in (PACKAGE_ROOT, SRC_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from sqlalchemy import select

from database.connection import SessionLocal
from database.langchain_pg import build_semantic_knowledge_document, get_semantic_knowledge_store
from database.models import SemanticKnowledge
from shared.logging import setup_logger

logger = setup_logger(__name__)


def migrate() -> int:
    session = SessionLocal()
    if session is None:
        raise RuntimeError("Database session factory is not initialized.")

    try:
        rows = session.execute(select(SemanticKnowledge)).scalars().all()
    finally:
        session.close()

    store = get_semantic_knowledge_store()
    documents = []
    ids = []
    for row in rows:
        documents.append(
            build_semantic_knowledge_document(
                key=row.key,
                category=row.category,
                description=row.description,
                value=row.value,
            )
        )
        ids.append(str(row.key))

    if documents:
        store.add_documents(documents, ids=ids)
    logger.info("Migrated %s semantic knowledge rows to PGVector.", len(documents))
    return len(documents)


if __name__ == "__main__":
    count = migrate()
    print(f"Migrated {count} semantic knowledge rows to PGVector.")
