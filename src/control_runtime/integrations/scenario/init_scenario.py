"""Scenario initialization entrypoint.

Usage:
  python -m control_runtime.integrations.scenario.init_scenario --graph-snapshot-id live-e1
"""

from .common import (
    cache_scenario,
    clear_cached_scenario,
    get_cached_control_scenario,
    get_cached_scenario,
    serialize_scenario_for_api,
)
from .runtime import (
    get_current_optimizer_scenario,
    get_current_scenario,
    get_initial_scenario,
    init_main,
    initialize_scenario,
)
from .ue_bootstrap import (
    rebuild_ue_related_tables_from_graph_snapshot,
    rebuild_ue_related_tables_from_latest_graph,
    sync_latest_flow_five_tuples_to_ue_context,
)
from .yaml_loader import deserialize_scenario_payload

__all__ = [
    "cache_scenario",
    "clear_cached_scenario",
    "deserialize_scenario_payload",
    "get_cached_control_scenario",
    "get_cached_scenario",
    "get_current_optimizer_scenario",
    "get_current_scenario",
    "get_initial_scenario",
    "init_main",
    "initialize_scenario",
    "rebuild_ue_related_tables_from_graph_snapshot",
    "rebuild_ue_related_tables_from_latest_graph",
    "serialize_scenario_for_api",
    "sync_latest_flow_five_tuples_to_ue_context",
]


if __name__ == "__main__":
    init_main()
