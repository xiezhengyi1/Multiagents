from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

from langchain_core.messages import BaseMessage, ToolMessage

from .token_budget import TokenBudget, TokenCounter


_TRUNCATION_SUFFIX = "\n... [truncated]"


def _count_unclosed(text: str) -> tuple[int, int]:
    """Return (open_braces, open_brackets) that are unclosed in *text*."""
    braces = 0
    brackets = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            braces += 1
        elif ch == "}":
            braces -= 1
        elif ch == "[":
            brackets += 1
        elif ch == "]":
            brackets -= 1
    return max(0, braces), max(0, brackets)


def _truncate_text_with_marker(text: str, max_chars: int) -> str:
    """Truncate to at most *max_chars*, trying to stay on a section boundary
    and keeping any JSON structure parseable."""
    if len(text) <= max_chars:
        return text

    candidate = text[:max_chars]
    # Prefer a newline boundary (section-based output) that is near the limit
    last_nl = candidate.rfind("\n")
    if last_nl > max_chars // 2:
        candidate = candidate[:last_nl].rstrip()

    stripped = candidate.strip()
    if stripped and stripped[0] in "{[":
        # Looks like JSON — close any unclosed braces / brackets
        braces, brackets = _count_unclosed(candidate)
        if braces or brackets:
            candidate = candidate + "}" * braces + "]" * brackets
        return candidate + _TRUNCATION_SUFFIX

    return candidate + _TRUNCATION_SUFFIX


@dataclass(frozen=True)
class ContextPolicy:
    """Single policy for limiting model-visible context growth."""

    default_tool_result_chars: int = 8000
    default_tool_result_tokens: int = 2000
    tool_result_char_limits: dict[str, int] = field(default_factory=dict)
    tool_result_token_limits: dict[str, int] = field(default_factory=dict)
    recent_tool_results: int = 2
    recent_tool_results_per_tool: int | None = None
    tool_history_keep_limits: dict[str, int] = field(default_factory=dict)

    def _char_limit(self, tool_name: str) -> int:
        return max(1, int(self.tool_result_char_limits.get(tool_name, self.default_tool_result_chars)))

    def _token_limit(self, tool_name: str, token_budget: TokenBudget | None) -> int:
        limit = max(1, int(self.tool_result_token_limits.get(tool_name, self.default_tool_result_tokens)))
        if token_budget is None:
            return limit
        pressure = token_budget.pressure()
        if pressure >= 0.8:
            return min(limit, max(256, limit // 2))
        if pressure >= 0.5:
            return min(limit, max(256, int(limit * 0.75)))
        return limit

    def compact_tool_result(
        self,
        tool_name: str,
        content: str,
        *,
        token_counter: TokenCounter | None = None,
        token_budget: TokenBudget | None = None,
    ) -> str:
        """Truncate a tool result to the character limit configured for *tool_name*.

        Token-level truncation is intentionally NOT applied here — per-tool call
        count limits and ``max_iterations`` are the primary backstop against
        context explosion.  Character truncation only fires on genuinely huge
        payloads and uses JSON-safe closing so downstream parsers can still
        consume the result.
        """
        compacted = str(content or "")
        char_limit = self._char_limit(tool_name)
        if len(compacted) > char_limit:
            return _truncate_text_with_marker(compacted, char_limit)
        return compacted

    def compact_tool_history(self, messages: Iterable[BaseMessage]) -> list[BaseMessage]:
        compacted = list(messages)
        keep_indexes = self._tool_history_keep_indexes(compacted)
        for index, message in enumerate(compacted):
            if isinstance(message, ToolMessage) and index not in keep_indexes:
                compacted[index] = message.model_copy(
                    update={
                        "content": (
                            '{"status":"compacted","note":"older tool result omitted; '
                            'call the tool again only if the missing detail is required"}'
                        )
                    }
                )
        return compacted

    def _tool_history_keep_indexes(self, messages: list[BaseMessage]) -> set[int]:
        indexes_by_tool: dict[str, list[int]] = {}
        for index, message in enumerate(messages):
            if isinstance(message, ToolMessage):
                tool_name = str(message.name or "unknown")
                indexes_by_tool.setdefault(tool_name, []).append(index)

        keep_indexes: set[int] = set()
        for tool_name, indexes in indexes_by_tool.items():
            keep_count = self._tool_history_keep_count(tool_name)
            if keep_count > 0:
                keep_indexes.update(indexes[-keep_count:])
        return keep_indexes

    def _tool_history_keep_count(self, tool_name: str) -> int:
        if tool_name in self.tool_history_keep_limits:
            return max(0, int(self.tool_history_keep_limits[tool_name]))
        if self.recent_tool_results_per_tool is not None:
            return max(0, int(self.recent_tool_results_per_tool))
        return max(0, int(self.recent_tool_results))

    def compact_text(
        self,
        text: str,
        *,
        max_chars: int,
        max_tokens: int | None = None,
        token_counter: TokenCounter | None = None,
    ) -> str:
        """Keep the newest structured sections while enforcing all configured budgets."""
        compacted = self._compact_sections_by_chars(str(text or ""), max(1, int(max_chars)))
        if token_counter is None or max_tokens is None:
            return compacted
        return self._compact_sections_by_tokens(
            compacted,
            max(1, int(max_tokens)),
            token_counter,
        )

    @staticmethod
    def _split_sections(text: str) -> list[str]:
        sections = text.split("\n[")
        return [section if index == 0 else "[" + section for index, section in enumerate(sections)]

    @classmethod
    def _compact_sections_by_chars(cls, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        keep: list[str] = []
        size = 0
        for section in reversed(cls._split_sections(text)):
            overhead = 1 if keep else 0
            if size + len(section) + overhead > max_chars:
                break
            keep.insert(0, section)
            size += len(section) + overhead
        return "\n".join(keep) if keep else text[-max_chars:]

    @classmethod
    def _compact_sections_by_tokens(
        cls,
        text: str,
        max_tokens: int,
        token_counter: TokenCounter,
    ) -> str:
        if token_counter.count(text) <= max_tokens:
            return text
        keep: list[str] = []
        tokens = 0
        newline_tokens = token_counter.count("\n")
        for section in reversed(cls._split_sections(text)):
            overhead = newline_tokens if keep else 0
            section_tokens = token_counter.count(section)
            if tokens + section_tokens + overhead > max_tokens:
                break
            keep.insert(0, section)
            tokens += section_tokens + overhead
        if keep:
            return "\n".join(keep)
        return cls._tail_within_tokens(text, max_tokens, token_counter)

    @staticmethod
    def _tail_within_tokens(text: str, max_tokens: int, token_counter: TokenCounter) -> str:
        low, high = 0, len(text)
        while low < high:
            midpoint = (low + high + 1) // 2
            if token_counter.count(text[-midpoint:]) <= max_tokens:
                low = midpoint
            else:
                high = midpoint - 1
        return text[-low:] if low else ""


__all__ = ["ContextPolicy"]
