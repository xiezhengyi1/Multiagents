from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "experiments" / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import paths
from experiments.scripts import common


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


def test_launch_experiments_accepts_qwen_flag(monkeypatch) -> None:
    from experiments.scripts import launch_experiments

    monkeypatch.setattr(sys, "argv", ["launch_experiments.py", "--qwen"])

    args = launch_experiments.parse_args()

    assert args.use_qwen is True


def test_run_method_forwards_qwen_to_single_agent_runner(tmp_path: Path, monkeypatch) -> None:
    from experiments.scripts import run_method

    monkeypatch.setattr(run_method, "RAW_RUN_DIR", tmp_path)
    monkeypatch.setattr(run_method, "SUMMARY_DIR", tmp_path)
    monkeypatch.setattr(run_method, "_build_user_inputs_path", lambda **_: tmp_path / "inputs.json")
    monkeypatch.setattr(run_method, "_resolve_case_count", lambda *_args, **_kwargs: 1)
    (tmp_path / "inputs.json").write_text(json.dumps({"records": [{"id": 1}]}), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> None:
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr(run_method.subprocess, "run", fake_run)

    run_method._run_single_agent(
        method_id="B2",
        experiment_id="E1",
        scenario_id="S2",
        snapshot_id="live-snap",
        start_index=1,
        use_qwen=True,
    )

    assert "--qwen" in captured["command"]


def test_run_single_agent_experiment_accepts_qwen_flag(monkeypatch) -> None:
    from experiments.scripts import run_single_agent_experiment

    monkeypatch.setattr(sys, "argv", ["run_single_agent_experiment.py", "--qwen"])

    args = run_single_agent_experiment.parse_args()

    assert args.use_qwen is True


def test_single_agent_orchestrator_passes_model_name(monkeypatch) -> None:
    from control_runtime.orchestrators import single_agent_orchestrator

    captured: dict[str, object] = {}

    class FakeSingleControlAgent:
        def __init__(self, **kwargs: object) -> None:
            captured["single_kwargs"] = kwargs
            self.model_name = kwargs.get("model_name", "")

    class FakePolicyDispatchAgent:
        def __init__(self, **kwargs: object) -> None:
            captured["dispatch_kwargs"] = kwargs

    monkeypatch.setattr(single_agent_orchestrator, "SingleControlAgent", FakeSingleControlAgent)
    monkeypatch.setattr(single_agent_orchestrator, "PolicyDispatchAgent", FakePolicyDispatchAgent)
    monkeypatch.setattr(single_agent_orchestrator, "ConflictResolutionTool", lambda: SimpleNamespace())
    monkeypatch.setattr(single_agent_orchestrator, "AssuranceDiagnosisTool", lambda: SimpleNamespace())
    single_agent_orchestrator.SingleAgentOrchestrator._startup_banner_printed = False

    single_agent_orchestrator.SingleAgentOrchestrator(
        max_rounds=1,
        single_model_name="qwen3-30b-a3b-instruct",
    )

    assert captured["single_kwargs"]["model_name"] == "qwen3-30b-a3b-instruct"


def test_workflow_summary_reports_average_elapsed_ms(tmp_path: Path) -> None:
    from experiments.scripts.run_workflow_experiment import summarize_run_results

    summary = summarize_run_results(
        [
            {"status": "error", "completed": False, "elapsed_ms": 120, "round_count": 0, "retry_count": 0},
            {"status": "error", "completed": False, "elapsed_ms": 280, "round_count": 0, "retry_count": 0},
        ],
        result_output=tmp_path / "runs.jsonl",
        workflow_output=tmp_path / "workflow.jsonl",
    )

    assert summary["avg_elapsed_ms"] == 200.0


def test_single_agent_summary_reports_average_elapsed_ms(tmp_path: Path) -> None:
    from experiments.scripts.run_single_agent_experiment import summarize_run_results

    summary = summarize_run_results(
        [
            {"status": "success", "completed": True, "elapsed_ms": 90},
            {"status": "error", "completed": False, "elapsed_ms": 110},
        ],
        result_output=tmp_path / "runs.jsonl",
    )

    assert summary["avg_elapsed_ms"] == 100.0


def test_aggregate_extracts_elapsed_ms_for_ledger_latency() -> None:
    from experiments.scripts.aggregate_results import _extract_elapsed_ms

    assert _extract_elapsed_ms({"elapsed_ms": 42.6}) == 42.6
    assert _extract_elapsed_ms({"elapsed_ms": "15.5"}) == 15.5
    assert _extract_elapsed_ms({"elapsed_ms": ""}) == ""


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
