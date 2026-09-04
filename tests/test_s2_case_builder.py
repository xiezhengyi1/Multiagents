from __future__ import annotations

from experiments.scripts.build_s2_cases import EXPECTED_TASK_IDS, build_s2_cases


def test_build_s2_cases_restores_original_inventory_and_metadata() -> None:
    cases = build_s2_cases(run_id="s2-live-test")

    assert len(cases) == 30
    assert tuple(case.metadata["task"]["task_id"] for case in cases) == EXPECTED_TASK_IDS
    assert len({case.episode_id for case in cases}) == 30
    assert all(case.metadata["scenario_id"] == "S2" for case in cases)
    assert cases[0].user_input == (
        "把 imsi-208930000000001 的 Remote_Drive_video_1 迁移到更低时延的切片，"
        "优先保证控制稳定性。"
    )
    assert cases[1].user_input.endswith("（supi: imsi-208930000000008）")
    assert cases[-1].metadata["task"]["success_criteria"] == (
        "跨域职责边界清晰，RFSP 不被误改"
    )
