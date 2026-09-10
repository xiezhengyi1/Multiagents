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

