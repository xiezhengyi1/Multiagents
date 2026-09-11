from __future__ import annotations

import hashlib
from collections import Counter

from experiments.scripts.build_s2k_topology_cases import (
    DEFAULT_SCENARIO_ROOT,
    build_s2k_topology_cases,
)


def test_topology_catalog_has_fixed_quota_and_scenario_level_partitions() -> None:
    cases = build_s2k_topology_cases(run_id="s2k-topology-test")

    assert len({case.user_input for case in cases}) == 1000
    assert len(cases) == 1000
    assert len({case.case_id for case in cases}) == 1000
    assert Counter(case.metadata["case_class"] for case in cases) == {
        "feasible": 600,
        "infeasible": 400,
    }
    assert Counter(case.metadata["partition"] for case in cases) == {
        "train": 700,
        "validation": 150,
        "test": 150,
    }
    assert set(Counter(case.metadata["scenario_id"] for case in cases).values()) == {50}
    assert {case.metadata["catalog_version"] for case in cases} == {
        "s2k-topology-v2-natural-input"
    }
    assert all(
        case.metadata["sft_disposition"] == "pending_execution" for case in cases
    )

    partitions_by_topology: dict[str, set[str]] = {}
    for case in cases:
        topology = case.metadata["network_topology_sha256"]
        partitions_by_topology.setdefault(topology, set()).add(
            case.metadata["partition"]
        )
    assert len(partitions_by_topology) == 20
    assert all(len(partitions) == 1 for partitions in partitions_by_topology.values())


def test_topology_catalog_bindings_and_hashes_match_scenario_files() -> None:
    cases = build_s2k_topology_cases(run_id="s2k-topology-test")
    by_scenario = {case.metadata["scenario_id"]: case for case in cases}

    for case in by_scenario.values():
        path = (
            DEFAULT_SCENARIO_ROOT
            / case.metadata["scenario_config_relative_path"].split("/")[-1]
        )
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == case.metadata["scenario_config_sha256"]
        )

    for case in cases:
        initial = case.metadata["scenario_initial_state"]
        binding = case.metadata["flow_binding"]
        authorized = {(item["dnn"], item["snssai"]) for item in initial["subscription"]}
        assert (binding["dnn"], binding["snssai"]) in authorized
        if case.metadata["operation_class"] == "ursp_existing_pdu":
            target = case.metadata["requested_route"]
            assert (target["dnn"], target["snssai"]) in authorized
        if case.metadata["operation_class"] == "hard_constraint_conflict":
            target = case.metadata["requested_route"]
            assert (target["dnn"], target["snssai"]) in authorized
        if case.metadata["operation_class"] == "subscription_entitlement_block":
            target = case.metadata["requested_route"]
            assert (target["dnn"], target["snssai"]) not in authorized


def test_topology_catalog_operation_mix_is_stable() -> None:
    cases = build_s2k_topology_cases(run_id="s2k-topology-test")
    assert Counter(case.metadata["operation_class"] for case in cases) == {
        "qos_same_pdu": 360,
        "ursp_existing_pdu": 120,
        "am_rfsp_only": 60,
        "multi_supi_staged": 40,
        "safe_qos_execution": 20,
        "subscription_entitlement_block": 200,
        "hard_constraint_conflict": 200,
    }


def test_topology_catalog_user_inputs_are_natural_and_keep_supi_only() -> None:
    cases = build_s2k_topology_cases(run_id="s2k-topology-test")

    for case in cases:
        bindings = case.metadata["flow_bindings"]
        assert not case.user_input.startswith("在 ")
        for binding in bindings:
            assert binding["supi"] in case.user_input
            for internal_value in (
                binding["flow_id"],
                binding["app_id"],
                binding["session_id"],
                binding["dnn"],
                binding["snssai"],
                binding["slice_id"],
            ):
                assert internal_value not in case.user_input
