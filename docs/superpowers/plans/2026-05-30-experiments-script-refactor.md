# Experiments Script Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce duplicated and mixed-purpose code under `experiments/` while preserving every existing CLI, argument, configuration path, output path, strict failure mode, and external import point.

**Architecture:** Keep the current CLI files as stable entry points. Consolidate path and file helpers in `experiments/paths.py` and `experiments/scripts/common.py`, then extract simulator lifecycle code from `launch_experiments.py` into `experiments/scripts/simulator.py`. Apply only small follow-up deduplications where tests prove unchanged behavior.

**Tech Stack:** Python 3.11+, `pytest`, standard library, existing `PyYAML`

---

## File Map

- Modify `experiments/paths.py`: canonical experiment path constants and scenario resolution.
- Modify `experiments/scripts/common.py`: compatibility re-exports plus minimal JSON, JSONL, CSV, environment, and project-script helpers.
- Create `experiments/scripts/simulator.py`: simulator and graph-snapshot lifecycle extracted from `launch_experiments.py`.
- Modify `experiments/scripts/launch_experiments.py`: matrix orchestration only.
- Modify `experiments/scripts/launch_guarded_system.py`: consume `simulator.py`.
- Modify `experiments/scripts/run_method.py`: consume common environment and script helpers.
- Modify `experiments/scripts/run_single_agent_experiment.py`: consume common output helpers.
- Modify `experiments/scripts/run_workflow_experiment.py`: consume common output helpers.
- Modify `experiments/scripts/aggregate_results.py`: consume common JSONL and CSV helpers.
- Modify `experiments/scripts/compute_thesis_metrics.py`: consume common JSONL, JSON, and CSV helpers.
- Modify input and profile scripts only where shared helpers remove direct duplication without changing algorithms.
- Create `tests/test_experiment_script_refactor.py`: compatibility and extraction regression tests.

### Task 1: Lock Compatibility With Failing Tests

**Files:**
- Create: `tests/test_experiment_script_refactor.py`
- Read: `experiments/configs/methods.json`
- Read: `experiments/paths.py`

- [ ] **Step 1: Write compatibility tests before changing production code**

```python
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
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_experiment_script_refactor.py -q`

Expected: FAIL because `append_jsonl` and `read_jsonl` do not exist yet.

- [ ] **Step 3: Commit the failing test**

```powershell
git add tests/test_experiment_script_refactor.py
git commit -m "test: lock experiments script compatibility"
```

### Task 2: Consolidate Paths and File Helpers

**Files:**
- Modify: `experiments/paths.py`
- Modify: `experiments/scripts/common.py`
- Test: `tests/test_experiment_script_refactor.py`

- [ ] **Step 1: Move all experiment-root constants into `experiments/paths.py`**

Add canonical constants:

```python
PROJECT_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
SCENARIO_ROOT = PACKAGE_ROOT / "scenarios"
TASK_ROOT = PACKAGE_ROOT / "tasks"
RESULTS_ROOT = PACKAGE_ROOT / "results"
LEDGER_ROOT = RESULTS_ROOT / "ledgers"
RAW_RUN_ROOT = RESULTS_ROOT / "raw_runs"
SUMMARY_ROOT = RESULTS_ROOT / "summaries"
```

- [ ] **Step 2: Re-export paths and add minimal file helpers in `common.py`**

Add helpers with these signatures:

```python
def load_json(path: Path) -> Dict[str, Any]: ...
def write_json(path: Path, payload: Any) -> None: ...
def read_jsonl(path: Path) -> List[Dict[str, Any]]: ...
def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None: ...
def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None: ...
def build_project_python_env() -> Dict[str, str]: ...
def run_project_python(script: Path, *args: str, cwd: Path = WORKSPACE_ROOT) -> None: ...
```

Implementation constraints:

```python
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")
```

- [ ] **Step 3: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_experiment_script_refactor.py -q`

Expected: PASS.

- [ ] **Step 4: Commit**

```powershell
git add experiments/paths.py experiments/scripts/common.py tests/test_experiment_script_refactor.py
git commit -m "refactor: centralize experiment script utilities"
```

### Task 3: Extract Simulator Lifecycle

**Files:**
- Create: `experiments/scripts/simulator.py`
- Modify: `experiments/scripts/launch_experiments.py`
- Modify: `experiments/scripts/launch_guarded_system.py`
- Test: `tests/test_experiment_script_refactor.py`

- [ ] **Step 1: Add a failing import-boundary test**

Append:

```python
def test_simulator_lifecycle_has_dedicated_module() -> None:
    from experiments.scripts import simulator

    assert callable(simulator.start_simulator_for_scenario)
    assert callable(simulator.stop_simulator)
    assert callable(simulator.restart_simulator_for_current_scenario)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_experiment_script_refactor.py::test_simulator_lifecycle_has_dedicated_module -q`

Expected: FAIL because `experiments.scripts.simulator` does not exist.

- [ ] **Step 3: Extract simulator functions without semantic edits**

Move the simulator and live-graph functions from `launch_experiments.py` into `simulator.py`. Rename only the public functions used outside the module:

```python
start_simulator_for_scenario = _start_simulator_for_scenario
restart_simulator_for_current_scenario = _restart_simulator_for_current_scenario
ensure_simulator_running = _ensure_simulator_running
stop_simulator = _stop_simulator
```

Keep private graph and probing helpers private. Update both entry points to import the public names from `simulator.py`.

- [ ] **Step 4: Run focused tests and CLI smoke checks**

Run:

```powershell
python -m pytest tests/test_experiment_script_refactor.py -q
python experiments/scripts/launch_experiments.py --help
python experiments/scripts/launch_guarded_system.py --help
```

Expected: PASS and both help commands exit `0`.

- [ ] **Step 5: Commit**

```powershell
git add experiments/scripts/simulator.py experiments/scripts/launch_experiments.py experiments/scripts/launch_guarded_system.py tests/test_experiment_script_refactor.py
git commit -m "refactor: isolate experiment simulator lifecycle"
```

### Task 4: Reuse Shared I/O and Process Helpers

**Files:**
- Modify: `experiments/scripts/run_method.py`
- Modify: `experiments/scripts/run_single_agent_experiment.py`
- Modify: `experiments/scripts/run_workflow_experiment.py`
- Modify: `experiments/scripts/aggregate_results.py`
- Modify: `experiments/scripts/compute_thesis_metrics.py`
- Modify: `experiments/scripts/build_user_inputs.py`
- Modify: `experiments/scripts/build_public_dataset_profiles.py`
- Test: `tests/test_compute_thesis_metrics.py`
- Test: `tests/test_iea_osa_contract_regressions.py`
- Test: `tests/test_experiment_script_refactor.py`

- [ ] **Step 1: Add failing tests for CSV and JSON writers**

Append:

```python
def test_common_writers_create_parent_directories(tmp_path: Path) -> None:
    from experiments.scripts.common import write_csv, write_json

    json_path = tmp_path / "nested" / "payload.json"
    csv_path = tmp_path / "nested" / "rows.csv"
    write_json(json_path, {"ok": True})
    write_csv(csv_path, [{"id": "1"}], ["id"])
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}
    assert csv_path.read_text(encoding="utf-8").splitlines() == ["id", "1"]
```

- [ ] **Step 2: Run the new test and verify RED if helper behavior is incomplete**

Run: `python -m pytest tests/test_experiment_script_refactor.py::test_common_writers_create_parent_directories -q`

Expected: PASS if Task 2 already implemented the exact contract; otherwise FAIL and complete the helper before proceeding.

- [ ] **Step 3: Replace duplicate local I/O helpers only where the shared helper is equivalent**

Required replacements:

```python
from experiments.scripts.common import append_jsonl, read_jsonl, write_csv, write_json
```

Keep domain-specific transforms such as metric field ordering, summary construction, and task filtering local. Delete only local wrappers made redundant by shared helpers.

- [ ] **Step 4: Replace `run_method.py` environment setup with `build_project_python_env()`**

Import:

```python
from experiments.scripts.common import build_project_python_env
```

Delete `_build_project_python_env` from `run_method.py` and update the scenario reset subprocess call.

- [ ] **Step 5: Run focused regression tests**

Run:

```powershell
python -m pytest tests/test_experiment_script_refactor.py tests/test_compute_thesis_metrics.py tests/test_iea_osa_contract_regressions.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add experiments/scripts tests/test_experiment_script_refactor.py
git commit -m "refactor: reuse experiment script infrastructure"
```

### Task 5: Verify Full Compatibility and Reduction

**Files:**
- Read: `experiments/README.md`
- Read: `README.md`
- Read: `experiments/configs/methods.json`
- Read: `experiments/scripts/*.py`

- [ ] **Step 1: Run all preserved CLI help checks**

Run:

```powershell
python -m pytest tests/test_experiment_script_refactor.py -q
```

Expected: PASS.

- [ ] **Step 2: Run configured input-generation smoke check**

Run:

```powershell
python experiments/scripts/build_user_inputs.py --experiment E1 --scenario S2
```

Expected: exit `0` and a generated scoped input file under `experiments/generated_inputs/`.

- [ ] **Step 3: Run the relevant complete test suite**

Run:

```powershell
python -m pytest tests/test_compute_thesis_metrics.py tests/test_iea_osa_contract_regressions.py tests/test_environment_agent.py tests/test_monitor_reentry_loop.py tests/test_experiment_script_refactor.py -q
```

Expected: PASS.

- [ ] **Step 4: Measure final script sizes and inspect diff**

Run:

```powershell
@'
from pathlib import Path
for path in sorted(Path("experiments/scripts").glob("*.py")):
    print(f"{path}: {len(path.read_text(encoding='utf-8').splitlines())}")
'@ | python -
git diff --stat HEAD~4..HEAD
git status --short
```

Expected: `launch_experiments.py` is substantially smaller; status contains no accidental modifications outside planned files and the user's pre-existing work.

- [ ] **Step 5: Commit any final documentation adjustment**

```powershell
git add experiments/README.md README.md
git commit -m "docs: describe streamlined experiment scripts"
```

Only run this commit when documentation actually needs adjustment.
