"""tools 对外稳定入口。

关键步骤：上层模块优先从本模块导入，避免深路径耦合。
"""

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

__all__ = [
    "normalize_app_input_with_ue_contexts",
    "validate_input_contract",
    "get_current_scenario",
    "get_initial_scenario",
    "cache_scenario",
    "get_cached_scenario",
    "clear_cached_scenario",
    "serialize_scenario_for_api",
    "deserialize_scenario_payload",
    "get_network_status",
    "run_joint_simulation",
    "update_ue_context_after_policy",
    "get_latest_session_context",
    "create_session_context",
    "update_session_context",
]


def __getattr__(name):
    # 兼容历史路径：允许 tools.db_tool / tools.pcf_tools 被动态访问
    if name in {"db_tool", "pcf_tools", "io_handler", "init_scenario", "ran_scheduler", "optimizer", "common"}:
        return importlib.import_module(f"tools.{name}")
    raise AttributeError(f"module 'tools' has no attribute '{name}'")
