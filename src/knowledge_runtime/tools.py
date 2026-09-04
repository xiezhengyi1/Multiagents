from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.tools import BaseTool, tool

from .contracts import KnowledgeDomain, KnowledgeResponse


class RagBackend(Protocol):
    def search(
        self,
        *,
        query: str,
        category: KnowledgeDomain | None,
        limit: int,
    ) -> KnowledgeResponse: ...

    def get_by_key(
        self,
        *,
        key: str,
        category: KnowledgeDomain | None,
    ) -> KnowledgeResponse: ...


@dataclass(frozen=True, slots=True)
class RagTools:
    search: BaseTool
    get_by_key: BaseTool

    @property
    def all(self) -> tuple[BaseTool, BaseTool]:
        return self.search, self.get_by_key


def create_rag_tools(backend: RagBackend) -> RagTools:
    @tool("search_semantic_knowledge")
    def search_semantic_knowledge(
        query: str,
        category: KnowledgeDomain | None = None,
        limit: int = 10,
    ) -> dict[str, object]:
        """Search an injected standards knowledge backend for one exact unresolved fact."""

        return backend.search(
            query=query,
            category=category,
            limit=limit,
        ).model_dump(mode="json")

    @tool("get_knowledge_by_key")
    def get_knowledge_by_key(
        key: str,
        category: KnowledgeDomain | None = None,
    ) -> dict[str, object]:
        """Fetch one exact standards object from an injected knowledge backend."""

        return backend.get_by_key(key=key, category=category).model_dump(mode="json")

    return RagTools(
        search=search_semantic_knowledge,
        get_by_key=get_knowledge_by_key,
    )


__all__ = ["KnowledgeDomain", "RagBackend", "RagTools", "create_rag_tools"]
