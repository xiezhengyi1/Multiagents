from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import tiktoken
from langchain_core.messages import BaseMessage


class TokenCounter:
    """Token-aware counting using tiktoken with boundary-aware truncation."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def encoding_name(self) -> str:
        return self._encoding_name

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def count_messages(self, messages: list[BaseMessage]) -> int:
        total = 0
        for message in messages:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        total += self.count(block)
                    elif isinstance(block, dict):
                        total += self.count(str(block.get("text", "")))
            total += 4
        total += 2
        return total

    def truncate_to_tokens(self, text: str, max_tokens: int, suffix: str = "") -> str:
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        candidate_tokens = tokens[:max_tokens]
        candidate = self._encoding.decode(candidate_tokens)
        boundary = self._find_truncation_boundary(candidate)
        return boundary + suffix

    @staticmethod
    def _find_truncation_boundary(candidate: str) -> str:
        last_nl = candidate.rfind("\n")
        if last_nl > len(candidate) * 0.6:
            return candidate[:last_nl]

        last_brace = candidate.rfind("}")
        last_bracket = candidate.rfind("]")
        last_structural = max(last_brace, last_bracket)
        if last_structural > len(candidate) * 0.6:
            close = candidate[: last_structural + 1]
            depth = close.count("{") - close.count("}")
            depth += close.count("[") - close.count("]")
            if depth <= 0:
                return close

        return candidate


@dataclass
class TokenBudget:
    total_limit: int = 128000
    reserved_output_tokens: int = 4096
    safety_margin_tokens: int = 1024
    system_prompt_tokens: int = 0
    tool_definitions_tokens: int = 0

    _consumed: dict[str, int] = field(default_factory=dict)

    @property
    def usable_limit(self) -> int:
        return max(
            0,
            self.total_limit
            - self.reserved_output_tokens
            - self.safety_margin_tokens
            - self.system_prompt_tokens
            - self.tool_definitions_tokens,
        )

    def record_tokens(self, agent: str, tokens: int) -> None:
        current = self._consumed.get(agent, 0)
        self._consumed[agent] = current + max(0, tokens)

    def consumed(self, agent: Optional[str] = None) -> int:
        if agent is not None:
            return self._consumed.get(agent, 0)
        return sum(self._consumed.values())

    def remaining(self, agent: Optional[str] = None) -> int:
        return max(0, self.usable_limit - self.consumed(agent))

    def pressure(self, agent: Optional[str] = None) -> float:
        if self.usable_limit <= 0:
            return 1.0
        return min(1.0, self.consumed(agent) / self.usable_limit)

    def can_fit(self, tokens: int, agent: str) -> bool:
        return self.remaining(agent) >= tokens


__all__ = ["TokenCounter", "TokenBudget"]
