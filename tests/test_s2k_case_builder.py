from __future__ import annotations

from collections import Counter

from experiments.scripts.build_s2k_cases import build_s2k_cases


def test_build_s2k_catalog_has_fixed_quota_and_gold_contract() -> None:
    cases = build_s2k_cases(run_id="s2k-test")

    assert len(cases) == 1000
    assert len({case.case_id for case in cases}) == 1000
    assert Counter(case.metadata["feasibility"] for case in cases) == {
        "feasible": 600,
        "infeasible": 400,
    }
    assert Counter(case.metadata["partition"] for case in cases) == {
        "train": 700,
        "validation": 150,
        "test": 150,
    }
    assert all(case.metadata["sft_disposition"] == "pending_execution" for case in cases)
    assert {case.metadata["catalog_version"] for case in cases} == {"s2k-v3"}
    assert len({case.metadata["scenario_context"]["id"] for case in cases}) == 10
    assert set(Counter(case.metadata["scenario_context"]["id"] for case in cases).values()) == {100}
    assert all(case.metadata["scenario_initial_state"] for case in cases)
    assert all(case.metadata["expected_terminal"] for case in cases)
    assert all(case.metadata["expected_agent_path"] for case in cases)


def test_s2k_variants_do_not_split_a_template_between_partitions() -> None:
    cases = build_s2k_cases(run_id="s2k-test")
    partitions_by_template: dict[int, set[str]] = {}
    for case in cases:
        key = case.metadata["template_ordinal"]
        partitions_by_template.setdefault(key, set()).add(case.metadata["partition"])

    assert all(len(partitions) == 1 for partitions in partitions_by_template.values())



def test_s2k_profile_preflight_contract() -> None:
    for case in build_s2k_cases(run_id="s2k-test"):
        family = case.case_id.split("-")[1]
        initial = case.metadata["scenario_initial_state"]
        binding = case.metadata["flow_binding"]
        target = case.metadata["requested_route"]["target_snssai"]
        pdu_snssais = {item.rsplit("/", maxsplit=1)[1] for item in initial["pdu_pool"]}

        assert binding["snssai"] in initial["subscription"]
        assert binding["snssai"] in pdu_snssais
        if family in {"F2", "F3"} and target is not None:
            assert target in initial["subscription"]
            assert target in pdu_snssais
        if family == "I1":
            assert target not in initial["subscription"]
        if family == "I2":
            assert target in initial["subscription"]
