from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MESSAGES_PATH = SRC / "shared" / "runtime" / "messages.py"
BASE_AGENT_PATH = SRC / "shared" / "agents" / "base.py"
CONTROL_CAPABILITIES_PATH = SRC / "control_runtime" / "tool_capabilities.py"
AGENT_CALLER_PATHS = (
    SRC / "control_runtime" / "agents" / "main" / "agent.py",
    SRC / "control_runtime" / "agents" / "grounding" / "agent.py",
    SRC / "control_runtime" / "agents" / "planning" / "agent.py",
    SRC / "control_runtime" / "agents" / "single" / "agent.py",
    SRC / "environment" / "agent.py",
)


class _NamedTool:
    def __init__(self, name: str, *, capabilities: object = None) -> None:
        self.name = name
        self.description = f"{name} description"
        self.args_schema = None
        self.capabilities = capabilities


class _Response(BaseModel):
    result: str = ""


def _load_source_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module_name: str,
    path: Path,
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _load_messages_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    langchain_core = ModuleType("langchain_core")
    langchain_messages = ModuleType("langchain_core.messages")

    class BaseMessage:
        pass

    class AIMessage(BaseMessage):
        pass

    class ToolMessage(BaseMessage):
        pass

    langchain_messages.AIMessage = AIMessage
    langchain_messages.BaseMessage = BaseMessage
    langchain_messages.ToolMessage = ToolMessage
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", langchain_messages)
    return _load_source_module(
        monkeypatch,
        module_name="_capability_messages_under_test",
        path=MESSAGES_PATH,
    )


def _load_base_agent_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    messages: ModuleType,
) -> ModuleType:
    shared_package = ModuleType("shared")
    runtime = ModuleType("shared.runtime")
    logging = ModuleType("shared.logging")
    dotenv = ModuleType("dotenv")
    langchain_openai = ModuleType("langchain_openai")

    class RuntimeCache:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.agent_name = ""

    class RuntimePersistence:
        pass

    class AgentRuntimeContext:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class StructuredToolLoop:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class JsonlTraceWriter:
        def __init__(self, agent_name: str) -> None:
            self.agent_name = agent_name

    class TracedStructuredAgent:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class ChatOpenAI:
        pass

    class ContextPolicy:
        pass

    class ArtifactEnvelope:
        pass

    runtime.ContextPolicy = ContextPolicy
    runtime.RuntimeCache = RuntimeCache
    runtime.RuntimePersistence = RuntimePersistence
    runtime.AgentRuntimeContext = AgentRuntimeContext
    runtime.StructuredToolLoop = StructuredToolLoop
    runtime.build_tool_specs = messages.build_tool_specs
    runtime.extract_tool_calls = lambda *_args, **_kwargs: []
    runtime.ArtifactEnvelope = ArtifactEnvelope
    runtime.JsonlTraceWriter = JsonlTraceWriter
    runtime.TracedStructuredAgent = TracedStructuredAgent
    logging.setup_logger = lambda *_args, **_kwargs: object()
    dotenv.load_dotenv = lambda *_args, **_kwargs: False
    langchain_openai.ChatOpenAI = ChatOpenAI

    monkeypatch.setitem(sys.modules, "shared", shared_package)
    monkeypatch.setitem(sys.modules, "shared.runtime", runtime)
    monkeypatch.setitem(sys.modules, "shared.logging", logging)
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai)
    return _load_source_module(
        monkeypatch,
        module_name="_capability_base_agent_under_test",
        path=BASE_AGENT_PATH,
    )


def test_build_tool_specs_uses_call_scoped_aliases_and_defensive_output_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _load_messages_module(monkeypatch)
    source_capabilities = ["runtime_evidence"]
    aliases = {"fetch_runtime": source_capabilities}
    specs = messages.build_tool_specs([_NamedTool("fetch_runtime")], capability_aliases=aliases)

    source_capabilities.append("mutated_after_build")
    specs[0]["capabilities"].append("mutated_trace")

    assert specs[0]["capabilities"] == ["runtime_evidence", "mutated_trace"]
    assert aliases == {"fetch_runtime": ["runtime_evidence", "mutated_after_build"]}
    assert messages.build_tool_specs([_NamedTool("fetch_runtime")], capability_aliases=aliases)[0]["capabilities"] == [
        "runtime_evidence",
        "mutated_after_build",
    ]


def test_build_tool_specs_prefers_tool_capabilities_and_defaults_to_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _load_messages_module(monkeypatch)
    specs = messages.build_tool_specs(
        [
            _NamedTool("tool_owned", capabilities=["tool_capability"]),
            _NamedTool("unmapped"),
        ],
        capability_aliases={"tool_owned": ["alias_capability"]},
    )

    assert specs[0]["capabilities"] == ["tool_capability"]
    assert specs[1]["capabilities"] == []


def test_control_mapping_is_read_only_and_import_has_no_registration_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _load_messages_module(monkeypatch)
    capabilities = _load_source_module(
        monkeypatch,
        module_name="_capability_control_mapping_under_test",
        path=CONTROL_CAPABILITIES_PATH,
    )

    assert not hasattr(messages, "_TOOL_CAPABILITY_ALIASES")
    assert not hasattr(messages, "register_capability_aliases")
    assert not hasattr(messages, "resolve_tool_capabilities")
    assert messages.build_tool_specs([_NamedTool("get_sm_ue_context")])[0]["capabilities"] == []
    assert messages.build_tool_specs(
        [_NamedTool("get_sm_ue_context")],
        capability_aliases=capabilities.CONTROL_RUNTIME_CAPABILITY_ALIASES,
    )[0]["capabilities"] == ["sm_ue_context"]
    with pytest.raises(TypeError):
        capabilities.CONTROL_RUNTIME_CAPABILITY_ALIASES["new_tool"] = ("new_capability",)


def test_base_agent_passes_capability_mapping_explicitly_to_trace_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _load_messages_module(monkeypatch)
    base_module = _load_base_agent_module(monkeypatch, messages=messages)

    class _FakeLlm:
        def bind_tools(self, tools: object) -> "_FakeLlm":
            self.bound_tools = list(tools)
            return self

    agent = object.__new__(base_module.BaseAgent)
    agent.agent_name = "isolated_agent"
    agent.model_name = "fake-model"
    agent.llm = _FakeLlm()
    traced = agent.create_json_agent(
        tools=[_NamedTool("fetch_runtime")],
        system_prompt="test",
        response_model=_Response,
        capability_aliases={"fetch_runtime": ("runtime_evidence",)},
    )

    assert traced.tool_specs == [
        {
            "name": "fetch_runtime",
            "description": "fetch_runtime description",
            "args_schema": None,
            "capabilities": ["runtime_evidence"],
        }
    ]


def test_every_agent_call_site_passes_a_capability_mapping() -> None:
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
        assert all(any(keyword.arg == "capability_aliases" for keyword in call.keywords) for call in calls), path
