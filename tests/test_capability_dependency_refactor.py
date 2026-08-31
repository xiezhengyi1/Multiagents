from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import BaseModel

from control_runtime.tool_capabilities import CONTROL_RUNTIME_CAPABILITY_ALIASES
from shared.runtime.messages import build_tool_catalog


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
AGENT_CALLER_PATHS = (
    SRC / "control_runtime" / "agents" / "main" / "agent.py",
    SRC / "control_runtime" / "agents" / "grounding" / "agent.py",
    SRC / "control_runtime" / "agents" / "planning" / "agent.py",
    SRC / "control_runtime" / "agents" / "single" / "agent.py",
    SRC / "environment" / "agent.py",
)


class _Input(BaseModel):
    query: str


class _NamedTool:
    def __init__(self, name: str, *, capabilities: list[str] | None = None) -> None:
        self.name = name
        self.description = f"{name} description"
        self.metadata = {
            "contract": {
                "side_effect": "read",
                "result_format": "json",
                "output_schema": {"type": "object"},
            },
            "capabilities": capabilities or [],
        }

    def get_input_schema(self) -> type[BaseModel]:
        return _Input


def test_tool_catalog_is_immutable_and_content_addressed() -> None:
    aliases = {"fetch_runtime": ["runtime_evidence"]}
    catalog = build_tool_catalog([_NamedTool("fetch_runtime")], capability_aliases=aliases)
    aliases["fetch_runtime"].append("mutated")

    assert catalog.tools[0].capabilities == []
    with pytest.raises(Exception):
        catalog.tools[0].capabilities = ["mutated"]
    rebuilt = build_tool_catalog(
        [_NamedTool("fetch_runtime", capabilities=["runtime_evidence"])],
        capability_aliases=aliases,
    )
    assert rebuilt.fingerprint != catalog.fingerprint


def test_tool_catalog_requires_explicit_contract_metadata() -> None:
    tool = _NamedTool("fetch_runtime")
    tool.metadata = {}
    with pytest.raises(ValueError, match="missing explicit contract metadata"):
        build_tool_catalog([tool])


def test_control_mapping_is_read_only() -> None:
    with pytest.raises(TypeError):
        CONTROL_RUNTIME_CAPABILITY_ALIASES["new_tool"] = ("new_capability",)


def test_every_agent_call_site_passes_capabilities_and_prompt_identity() -> None:
    for path in AGENT_CALLER_PATHS:
        module = ast.parse(path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_json_agent"
        ]
        assert calls, path
        for call in calls:
            keyword_names = {keyword.arg for keyword in call.keywords}
            assert {"capability_aliases", "prompt_name", "prompt_version"} <= keyword_names, path
