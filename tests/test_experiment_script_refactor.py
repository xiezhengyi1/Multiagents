from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from experiments import paths
from experiments.scripts import common


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "experiments" / "scripts"


def test_common_reexports_canonical_experiment_paths() -> None:
    assert common.EXPERIMENT_ROOT == paths.PACKAGE_ROOT
    assert common.CONFIG_ROOT == paths.CONFIG_ROOT
    assert common.GENERATED_INPUT_ROOT == paths.GENERATED_INPUT_ROOT


def test_common_jsonl_helpers_round_trip_records(tmp_path: Path) -> None:
    from experiments.scripts.common import append_jsonl, read_jsonl

    path = tmp_path / "records.jsonl"
    append_jsonl(path, {"id": 1})
    append_jsonl(path, {"id": 2})
    assert read_jsonl(path) == [{"id": 1}, {"id": 2}]


def test_common_writers_create_parent_directories(tmp_path: Path) -> None:
    from experiments.scripts.common import write_csv, write_json

    json_path = tmp_path / "nested" / "payload.json"
    csv_path = tmp_path / "nested" / "rows.csv"
    write_json(json_path, {"ok": True})
    write_csv(csv_path, [{"id": "1"}], ["id"])
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}
    assert csv_path.read_text(encoding="utf-8").splitlines() == ["id", "1"]


def test_simulator_lifecycle_has_dedicated_module() -> None:
    from experiments.scripts import simulator

    assert callable(simulator.start_simulator_for_scenario)
    assert callable(simulator.stop_simulator)
    assert callable(simulator.restart_simulator_for_current_scenario)


def test_aggregate_task_map_loads_canonical_catalog() -> None:
    from experiments.scripts.aggregate_results import _task_map

    assert len(_task_map()) == 20


def test_launch_failure_recorder_appends_jsonl(tmp_path: Path, monkeypatch) -> None:
    from experiments.scripts import launch_experiments
    from experiments.scripts.common import read_jsonl

    path = tmp_path / "failed.jsonl"
    monkeypatch.setattr(launch_experiments, "FAILED_RUNS_PATH", path)
    launch_experiments._append_failed_run({"stage": "smoke"})
    assert read_jsonl(path) == [{"stage": "smoke"}]


def test_configured_runner_paths_still_exist() -> None:
    payload = json.loads((paths.CONFIG_ROOT / "methods.json").read_text(encoding="utf-8"))
    for method in payload["methods"]:
        assert (PROJECT_ROOT / method["runner"]).is_file()


def test_preserved_cli_entry_points_offer_help() -> None:
    for name in [
        "aggregate_results.py",
        "build_user_inputs.py",
        "compute_thesis_metrics.py",
        "generate_user_inputs.py",
        "launch_experiments.py",
        "launch_guarded_system.py",
        "run_method.py",
        "run_single_agent_experiment.py",
        "run_workflow_experiment.py",
    ]:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_ROOT / name), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
