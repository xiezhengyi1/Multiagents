from __future__ import annotations

from dataclasses import dataclass, field

from .token_budget import TokenCounter


@dataclass(frozen=True)
class ContextPolicy:
    """Explicit limits for model-visible context and tool-result contracts."""

    default_tool_result_chars: int = 8000
    tool_result_char_limits: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.default_tool_result_chars < 1:
            raise ValueError("default_tool_result_chars must be >= 1")
        invalid = {
            name: limit
            for name, limit in self.tool_result_char_limits.items()
            if limit < 1
        }
        if invalid:
            raise ValueError(f"tool_result_char_limits must be >= 1: {invalid}")

    def _char_limit(self, tool_name: str) -> int:
        return self.tool_result_char_limits.get(tool_name, self.default_tool_result_chars)

    def validate_tool_result(self, tool_name: str, content: str) -> str:
        """Require tools to emit bounded results without altering model-visible evidence."""

        char_limit = self._char_limit(tool_name)
        if len(content) > char_limit:
            raise ValueError(
                f"tool {tool_name} result exceeds its contract limit: "
                f"{len(content)} > {char_limit} characters"
            )
        return content

    def compact_text(
        self,
        text: str,
        *,
        max_chars: int,
        max_tokens: int | None = None,
        token_counter: TokenCounter | None = None,
    ) -> str:
        """Keep the newest structured sections while enforcing all configured budgets."""
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        compacted = self._compact_sections_by_chars(str(text or ""), max_chars)
        if token_counter is None or max_tokens is None:
            return compacted
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        return self._compact_sections_by_tokens(
            compacted,
            max_tokens,
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
