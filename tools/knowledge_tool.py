from __future__ import annotations

import json
from typing import Optional

from langchain.tools import ToolRuntime, tool

from agent_runtime import AgentRuntimeContext
from database.langchain_pg import get_semantic_knowledge_store


def _log_prefix(runtime: ToolRuntime[AgentRuntimeContext] = None) -> str:
    if runtime is None:
        return "[knowledge_tool]"
    ctx = runtime.context
    return (
        f"[knowledge_tool][agent={ctx.agent_name}]"
        f"[session={ctx.session_id}]"
        f"[snapshot={ctx.snapshot_id}]"
    )


def _format_docs(docs) -> str:
    if not docs:
        return ""

    output = []
    for doc in docs:
        metadata = doc.metadata if isinstance(getattr(doc, "metadata", None), dict) else {}
        try:
            value_str = metadata.get("value_json") or json.dumps(metadata.get("value"), ensure_ascii=False)
        except Exception:
            value_str = str(metadata.get("value_json") or metadata.get("value") or "")

        output.append(
            (
                f"Key: {metadata.get('key', '')}\n"
                f"Category: {metadata.get('category', '')}\n"
                f"Description: {metadata.get('description', '')}\n"
                f"Value: {value_str}\n"
            )
        )
    return "\n---\n".join(output)


@tool
def search_semantic_knowledge(
    query: str,
    category: Optional[str] = None,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Search for domain knowledge using semantic retrieval over the PGVector knowledge collection.
    """
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return "No relevant knowledge found for an empty query."

    try:
        store = get_semantic_knowledge_store()
        search_kwargs = {"k": 3}
        normalized_category = str(category or "").strip()
        if normalized_category:
            search_kwargs["filter"] = {"category": normalized_category}
        retriever = store.as_retriever(search_kwargs=search_kwargs)
        docs = retriever.invoke(normalized_query)
    except Exception as exc:
        return f"{_log_prefix(runtime)} Error executing vector search: {exc}"

    formatted = _format_docs(docs)
    if not formatted:
        return f"No relevant knowledge found for '{normalized_query}'."
    return formatted


@tool
def get_knowledge_by_key(
    key: str,
    runtime: ToolRuntime[AgentRuntimeContext] = None,
) -> str:
    """
    Retrieve specific knowledge by exact metadata key first, then fall back to semantic search.
    """
    normalized_key = str(key or "").strip()
    if not normalized_key:
        return "Knowledge item not found for key: "

    try:
        store = get_semantic_knowledge_store()
        exact_docs = store.get_by_ids([normalized_key])
        if exact_docs:
            return exact_docs[0].metadata.get("value_json", "")

        approx_docs = store.similarity_search(normalized_key, k=1)
    except Exception as exc:
        return f"{_log_prefix(runtime)} Error retrieving key {normalized_key}: {exc}"

    if approx_docs:
        doc = approx_docs[0]
        return f"[Approximate Match: {doc.metadata.get('key', '')}] {doc.metadata.get('value_json', '')}"
    return f"Knowledge item not found for key: {normalized_key}"


__all__ = ["get_knowledge_by_key", "search_semantic_knowledge"]
