from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from openai import OpenAI

from database.connection import DATABASE_URL, RAW_DATABASE_URL


SEMANTIC_KNOWLEDGE_COLLECTION = "semantic_knowledge_docs"
PCF_AM_POLICY_CLAUSES_COLLECTION = "pcf_am_policy_clauses_r18"
PCF_AM_POLICY_SCHEMA_COLLECTION = "pcf_am_policy_schema_r18"
PCF_SM_POLICY_CLAUSES_COLLECTION = "pcf_sm_policy_clauses_r18"
PCF_SM_POLICY_SCHEMA_COLLECTION = "pcf_sm_policy_schema_r18"
PCF_URSP_CLAUSES_COLLECTION = "pcf_ursp_clauses_r18"
PCF_URSP_SCHEMA_COLLECTION = "pcf_ursp_schema_r18"
PCF_POLICY_GLOSSARY_COLLECTION = "pcf_policy_glossary_r18"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
MAX_EMBED_INPUT_LENGTH = 8192
DEFAULT_EMBED_BATCH_SIZE = 8
DEFAULT_EMBED_MAX_WORKERS = 4


def _get_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")


def _get_base_url() -> Optional[str]:
    configured = str(os.getenv("OPENAI_BASE_URL", "")).strip()
    if configured:
        return configured
    if os.getenv("DASHSCOPE_API_KEY"):
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return None


def get_embedding_model_name() -> str:
    return str(os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)).strip() or DEFAULT_EMBEDDING_MODEL


def get_embedding_dimensions() -> int:
    raw_value = str(os.getenv("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS)).strip()
    try:
        return int(raw_value)
    except ValueError:
        return DEFAULT_EMBEDDING_DIMENSIONS


def get_embedding_batch_size() -> int:
    raw_value = str(os.getenv("EMBEDDING_BATCH_SIZE", DEFAULT_EMBED_BATCH_SIZE)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        value = DEFAULT_EMBED_BATCH_SIZE
    return max(1, value)


def get_embedding_max_workers() -> int:
    raw_value = str(os.getenv("EMBEDDING_MAX_WORKERS", DEFAULT_EMBED_MAX_WORKERS)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        value = DEFAULT_EMBED_MAX_WORKERS
    return max(1, value)


class CompatibleOpenAIEmbeddings(Embeddings):
    """Repository-local embeddings adapter using the OpenAI-compatible SDK directly."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        resolved_api_key = str(api_key or _get_api_key() or "").strip()
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY or DASHSCOPE_API_KEY is required for embeddings.")

        self.model = str(model or get_embedding_model_name()).strip() or DEFAULT_EMBEDDING_MODEL
        self.dimensions = int(dimensions or get_embedding_dimensions())
        self.api_key = resolved_api_key
        self.base_url = str(base_url or _get_base_url() or "").strip() or None
        self.client = client or OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.batch_size = get_embedding_batch_size()
        self.max_workers = get_embedding_max_workers()

    def _normalize_text(self, text: str) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            raise ValueError("Embedding input must not be empty.")
        if len(normalized) > MAX_EMBED_INPUT_LENGTH:
            raise ValueError(
                f"Embedding input length {len(normalized)} exceeds provider limit {MAX_EMBED_INPUT_LENGTH}. "
                "Rebuild the corpus with provider-safe chunking."
            )
        return normalized

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        normalized_inputs = [self._normalize_text(text) for text in texts]
        if not normalized_inputs:
            return []

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": normalized_inputs,
        }
        if self.dimensions > 0:
            request_kwargs["dimensions"] = self.dimensions

        response = self.client.embeddings.create(**request_kwargs)
        if not response.data or len(response.data) != len(normalized_inputs):
            raise RuntimeError("Embedding provider returned no vectors.")
        return [list(item.embedding) for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        normalized_inputs = [self._normalize_text(text) for text in texts]
        if not normalized_inputs:
            return []
        if len(normalized_inputs) == 1:
            return self._embed_batch(normalized_inputs)

        batches = [
            (start_index, normalized_inputs[start_index : start_index + self.batch_size])
            for start_index in range(0, len(normalized_inputs), self.batch_size)
        ]
        results: list[Optional[list[float]]] = [None] * len(normalized_inputs)

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as executor:
            futures = [executor.submit(self._embed_batch, batch_texts) for _, batch_texts in batches]
            for (start_index, batch_texts), future in zip(batches, futures):
                batch_vectors = future.result()
                if len(batch_vectors) != len(batch_texts):
                    raise RuntimeError("Embedding batch result count does not match input count.")
                for offset, vector in enumerate(batch_vectors):
                    results[start_index + offset] = vector

        if any(vector is None for vector in results):
            raise RuntimeError("Embedding results are incomplete.")
        return [vector for vector in results if vector is not None]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


def get_openai_embeddings() -> CompatibleOpenAIEmbeddings:
    return CompatibleOpenAIEmbeddings()


def get_pgvector_connection() -> str:
    return DATABASE_URL


def get_pgvector_store(
    *,
    collection_name: str,
    embeddings: Optional[Embeddings] = None,
) -> PGVector:
    normalized = str(collection_name or "").strip()
    if not normalized:
        raise ValueError("collection_name is required.")
    return PGVector(
        embeddings=embeddings or get_openai_embeddings(),
        collection_name=normalized,
        connection=get_pgvector_connection(),
        embedding_length=get_embedding_dimensions(),
        use_jsonb=True,
    )


def get_semantic_knowledge_store(
    *,
    embeddings: Optional[Embeddings] = None,
    collection_name: str = SEMANTIC_KNOWLEDGE_COLLECTION,
) -> PGVector:
    return get_pgvector_store(
        collection_name=collection_name,
        embeddings=embeddings,
    )


def ensure_pgvector_collection(
    *,
    collection_name: str,
    embeddings: Optional[Embeddings] = None,
) -> PGVector:
    vector_store = get_pgvector_store(
        collection_name=collection_name,
        embeddings=embeddings,
    )
    vector_store.create_tables_if_not_exists()
    vector_store.create_collection()
    return vector_store


def rebuild_pgvector_collection(
    *,
    collection_name: str,
    embeddings: Optional[Embeddings] = None,
) -> PGVector:
    vector_store = ensure_pgvector_collection(
        collection_name=collection_name,
        embeddings=embeddings,
    )
    vector_store.delete_collection()
    vector_store = get_pgvector_store(
        collection_name=collection_name,
        embeddings=embeddings,
    )
    vector_store.create_collection()
    return vector_store


def ensure_semantic_knowledge_collection(
    *,
    store: Optional[PGVector] = None,
) -> PGVector:
    return store or ensure_pgvector_collection(collection_name=SEMANTIC_KNOWLEDGE_COLLECTION)


def setup_langgraph_postgres(
    *,
    database_url: Optional[str] = None,
    embeddings: Optional[Embeddings] = None,
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


def build_pgvector_document(
    *,
    page_content: str,
    metadata: Dict[str, Any],
) -> Document:
    normalized_content = str(page_content or "").strip()
    if not normalized_content:
        raise ValueError("page_content must not be empty.")
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary.")
    return Document(page_content=normalized_content, metadata=metadata)


__all__ = [
    "CompatibleOpenAIEmbeddings",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBED_BATCH_SIZE",
    "DEFAULT_EMBED_MAX_WORKERS",
    "PCF_AM_POLICY_CLAUSES_COLLECTION",
    "PCF_AM_POLICY_SCHEMA_COLLECTION",
    "PCF_POLICY_GLOSSARY_COLLECTION",
    "PCF_SM_POLICY_CLAUSES_COLLECTION",
    "PCF_SM_POLICY_SCHEMA_COLLECTION",
    "PCF_URSP_CLAUSES_COLLECTION",
    "PCF_URSP_SCHEMA_COLLECTION",
    "SEMANTIC_KNOWLEDGE_COLLECTION",
    "build_pgvector_document",
    "build_semantic_knowledge_document",
    "ensure_semantic_knowledge_collection",
    "ensure_pgvector_collection",
    "get_embedding_dimensions",
    "get_embedding_batch_size",
    "get_embedding_max_workers",
    "get_embedding_model_name",
    "get_openai_embeddings",
    "get_pgvector_connection",
    "get_pgvector_store",
    "get_semantic_knowledge_store",
    "rebuild_pgvector_collection",
    "setup_langgraph_postgres",
]
