"""Stable public entry points for tool-layer helpers."""

import importlib


def get_current_scenario():
    from .init_scenario import get_current_scenario as _impl
    return _impl()


def get_initial_scenario():
    from .init_scenario import get_initial_scenario as _impl
    return _impl()


def cache_scenario(apps, slices, nodes):
    from .init_scenario import cache_scenario as _impl
    return _impl(apps, slices, nodes)


def get_cached_scenario():
    from .init_scenario import get_cached_scenario as _impl
    return _impl()


def clear_cached_scenario():
    from .init_scenario import clear_cached_scenario as _impl
    return _impl()


def serialize_scenario_for_api(apps, slices, nodes):
    from .init_scenario import serialize_scenario_for_api as _impl
    return _impl(apps, slices, nodes)


def deserialize_scenario_payload(payload):
    from .init_scenario import deserialize_scenario_payload as _impl
    return _impl(payload)


def get_network_status(*args, **kwargs):
    from .network_status import get_network_status as _impl
    return _impl(*args, **kwargs)


def update_ue_context_after_policy(*args, **kwargs):
    from .db_tool import upsert_ue_context as _impl
    return _impl(*args, **kwargs)


def get_latest_session_context(*args, **kwargs):
    from .db_tool import get_latest_session_context as _impl
    return _impl(*args, **kwargs)


def create_session_context(*args, **kwargs):
    from .db_tool import create_session_context as _impl
    return _impl(*args, **kwargs)


def update_session_context(*args, **kwargs):
    from .db_tool import update_session_context as _impl
    return _impl(*args, **kwargs)


def get_latest_snapshot_metadata(*args, **kwargs):
    from .db_tool import get_latest_snapshot_metadata as _impl
    return _impl(*args, **kwargs)


def get_snapshot_data_by_id(*args, **kwargs):
    from .db_tool import get_snapshot_data_by_id as _impl
    return _impl(*args, **kwargs)


from .wrapper_think import think_tool as think
from .pcf_tools import get_ue_context
from .pcf_tools import get_ue_flow_catalog
from .pcf_tools import search_flow_targets_by_name
from .knowledge_tool import search_semantic_knowledge
from .knowledge_tool import get_knowledge_by_key
from .user_interaction_tool import ask_user_clarification

__all__ = [
    "get_current_scenario",
    "get_initial_scenario",
    "cache_scenario",
    "get_cached_scenario",
    "clear_cached_scenario",
    "serialize_scenario_for_api",
    "deserialize_scenario_payload",
    "get_network_status",
    "update_ue_context_after_policy",
    "get_latest_session_context",
    "get_latest_snapshot_metadata",
    "get_snapshot_data_by_id",
    "create_session_context",
    "update_session_context",
    "think",
    "get_ue_context",
    "get_ue_flow_catalog",
    "search_flow_targets_by_name",
    "search_semantic_knowledge",
    "get_knowledge_by_key",
    "ask_user_clarification",
]


def __getattr__(name):
    if name in {"db_tool", "pcf_tools", "init_scenario", "optimizer", "common"}:
        return importlib.import_module(f"tools.{name}")
    raise AttributeError(f"module 'tools' has no attribute '{name}'")
