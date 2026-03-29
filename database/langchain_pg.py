from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from database.connection import DATABASE_URL, RAW_DATABASE_URL


SEMANTIC_KNOWLEDGE_COLLECTION = "semantic_knowledge_docs"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_EMBEDDING_DIMENSIONS = 1024


def _get_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")


def get_embedding_model_name() -> str:
    return str(os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)).strip() or DEFAULT_EMBEDDING_MODEL


def get_embedding_dimensions() -> int:
    raw_value = str(os.getenv("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS)).strip()
    try:
        return int(raw_value)
    except ValueError:
        return DEFAULT_EMBEDDING_DIMENSIONS


def get_openai_embeddings() -> OpenAIEmbeddings:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or DASHSCOPE_API_KEY is required for embeddings.")
    return OpenAIEmbeddings(
        model=get_embedding_model_name(),
        dimensions=get_embedding_dimensions(),
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def get_pgvector_connection() -> str:
    return DATABASE_URL


def get_semantic_knowledge_store(
    *,
    embeddings: Optional[OpenAIEmbeddings] = None,
    collection_name: str = SEMANTIC_KNOWLEDGE_COLLECTION,
) -> PGVector:
    return PGVector(
        embeddings=embeddings or get_openai_embeddings(),
        collection_name=collection_name,
        connection=get_pgvector_connection(),
        embedding_length=get_embedding_dimensions(),
        use_jsonb=True,
    )


def ensure_semantic_knowledge_collection(
    *,
    store: Optional[PGVector] = None,
) -> PGVector:
    vector_store = store or get_semantic_knowledge_store()
    vector_store.create_tables_if_not_exists()
    vector_store.create_collection()
    return vector_store


def setup_langgraph_postgres(
    *,
    database_url: Optional[str] = None,
    embeddings: Optional[OpenAIEmbeddings] = None,
) -> None:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore

    raw_url = str(database_url or RAW_DATABASE_URL).strip()
    embedding_model = embeddings or get_openai_embeddings()

    with PostgresSaver.from_conn_string(raw_url) as saver:
        saver.setup()

    with PostgresStore.from_conn_string(
        raw_url,
        index={
            "dims": get_embedding_dimensions(),
            "embed": embedding_model,
            "fields": ["content"],
        },
    ) as store:
        store.setup()


def build_semantic_knowledge_document(
    *,
    key: str,
    category: Optional[str],
    description: Optional[str],
    value: Any,
) -> Document:
    details = ""
    if isinstance(value, dict):
        primitive_items = [f"{name}: {item}" for name, item in value.items() if isinstance(item, (str, int, float, bool))]
        if primitive_items:
            details = ". Details: " + ", ".join(primitive_items)
    elif value not in (None, ""):
        details = f". Details: {value}"

    page_content = (
        f"Key: {key}. "
        f"Category: {category or 'unknown'}. "
        f"Description: {description or ''}{details}"
    ).strip()
    metadata: Dict[str, Any] = {
        "key": str(key),
        "category": str(category or "").strip() or None,
        "description": str(description or "").strip() or None,
        "value_json": json.dumps(value, ensure_ascii=False),
    }
    return Document(page_content=page_content, metadata=metadata)


__all__ = [
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "DEFAULT_EMBEDDING_MODEL",
    "SEMANTIC_KNOWLEDGE_COLLECTION",
    "build_semantic_knowledge_document",
    "ensure_semantic_knowledge_collection",
    "get_embedding_dimensions",
    "get_embedding_model_name",
    "get_openai_embeddings",
    "get_pgvector_connection",
    "get_semantic_knowledge_store",
    "setup_langgraph_postgres",
]
