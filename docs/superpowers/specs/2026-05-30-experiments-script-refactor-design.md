# Experiments Script Refactor Design

## Goal

Refactor the mixed-purpose scripts under `experiments/` into the smallest clear implementation that preserves current behavior. The result must keep the existing command-line entry points, arguments, configuration formats, output paths, and externally imported helpers working.

## Current State

The experiment layer currently exposes these script families:

- Experiment orchestration:
  - `experiments/scripts/launch_experiments.py`
  - `experiments/scripts/run_method.py`
  - `experiments/scripts/run_workflow_experiment.py`
  - `experiments/scripts/run_single_agent_experiment.py`
  - `experiments/scripts/launch_guarded_system.py`
- Input generation:
  - `experiments/scripts/build_user_inputs.py`
  - `experiments/scripts/generate_user_inputs.py`
- Result processing:
  - `experiments/scripts/aggregate_results.py`
  - `experiments/scripts/compute_thesis_metrics.py`
- Public dataset profile generation:
  - `experiments/scripts/build_public_dataset_profiles.py`

The largest concentration of mixed responsibilities is `launch_experiments.py`: it owns matrix selection, child-process execution, simulator lifecycle, graph snapshot management, progress reporting, failure recording, and final aggregation. Several other scripts repeat path definitions and JSON, JSONL, CSV, and subprocess handling.

## Compatibility Contract

The refactor preserves:

1. Every existing script file as a runnable CLI entry point.
2. Existing CLI argument names and their meaning.
3. `experiments/configs/*.json` formats and runner paths in `methods.json`.
4. Existing generated-input, raw-run, ledger, summary, metric, and scenario paths.
5. Helpers imported by tests or other modules, including `aggregate_results._extract_task_id`.
6. Existing strict validation behavior: missing scenarios, snapshots, task references, and required files continue to fail clearly instead of silently falling back.
7. Current uncommitted monitoring and environment-agent work. The refactor may integrate with it but must not overwrite or revert it.

## Chosen Approach

Use thin CLI entry points backed by a small number of shared modules. This keeps existing commands stable while removing repeated infrastructure code and separating simulator concerns from experiment selection.

Rejected alternatives:

- A single new CLI with subcommands would reduce file count but break existing commands and configured runner paths.
- A deep package hierarchy would make responsibilities explicit but add more files and indirection than this repository needs.

## Target Structure

### `experiments/paths.py`

Own all experiment path constants and scenario-path resolution. `experiments/scripts/common.py` imports and re-exports these values where needed for compatibility instead of defining a second root hierarchy.

### `experiments/scripts/common.py`

Own small reusable infrastructure helpers:

- resolve the project Python executable;
- construct the project-aware Python environment;
- load and write JSON mappings;
- read and append JSONL records;
- write CSV rows;
- run project Python scripts.

Helpers remain intentionally small. Domain decisions stay in their calling scripts.

### `experiments/scripts/simulator.py`

Own simulator and live-graph lifecycle operations extracted from `launch_experiments.py`:

- resolve simulator paths and database URL;
- run graph-admin commands;
- test, delete, and prune graph snapshots;
- start, probe, wait for, restart, and stop the simulator;
- stream and inspect simulator logs.

`launch_experiments.py` and `launch_guarded_system.py` consume this module. Simulator behavior and error messages remain equivalent.

### Existing CLI Scripts

Existing scripts keep argument parsing and high-level orchestration:

- `launch_experiments.py`: select matrix entries, invoke input generation and method runs, record failures, aggregate results.
- `run_method.py`: select the configured runner, bind a scenario snapshot, invoke the runner, append the run ledger.
- `run_workflow_experiment.py` and `run_single_agent_experiment.py`: execute cases and produce raw JSONL plus summary JSON.
- input, metric, aggregation, and public-dataset scripts: retain their domain algorithms while using shared I/O helpers.

## Data Flow

1. `launch_experiments.py` filters matrix entries by CLI arguments.
2. It uses `simulator.py` to establish the scenario-specific simulator and graph snapshot.
3. It invokes `build_user_inputs.py` for the selected experiment and scenario.
4. It invokes `run_method.py`, which selects a runner from `methods.json`.
5. The chosen runner writes incremental raw JSONL and a summary JSON file.
6. `run_method.py` appends the run ledger and emits its machine-readable JSON response.
7. `launch_experiments.py` optionally invokes `aggregate_results.py`.
8. `compute_thesis_metrics.py` remains an explicit post-processing command for thesis metrics.

## Error Handling

- Shared helpers validate root JSON and YAML types.
- Child-process wrappers preserve non-zero exit behavior and captured output needed for diagnostics.
- Simulator startup failures continue to surface log tails.
- Missing input files, scenario IDs, graph snapshots, and dataset source files remain explicit failures.
- Batch orchestration records individual failures and proceeds where the current behavior already permits continuation.

## Testing Strategy

Verification is layered:

1. Add focused tests for shared I/O, path, and simulator wrapper behavior where extraction changes ownership.
2. Keep existing unit tests passing, especially:
   - `tests/test_compute_thesis_metrics.py`
   - `tests/test_iea_osa_contract_regressions.py`
   - `tests/test_environment_agent.py`
   - `tests/test_monitor_reentry_loop.py`
3. Run CLI `--help` smoke checks for every preserved entry point.
4. Run input-generation smoke checks against a real configured scenario.
5. Compare script inventory, `methods.json` runner paths, and output-path constants before and after refactoring.
6. Measure Python line counts before and after. The target is a meaningful reduction in duplicated code and a substantial reduction of mixed responsibilities in `launch_experiments.py`, not an arbitrary percentage.

## Migration Sequence

1. Add regression tests for compatibility-critical helpers and CLI entry points.
2. Consolidate path and I/O helpers.
3. Extract simulator lifecycle code.
4. Simplify orchestration scripts to consume shared helpers.
5. Simplify runner output handling.
6. Remove remaining duplicate I/O from result and input scripts where this shortens code without obscuring algorithms.
7. Run the complete verification set and inspect the final diff for unrelated changes.

## Non-Goals

- Changing experiment semantics, matrices, task definitions, scenarios, or thesis metric formulas.
- Renaming or removing current CLI scripts.
- Rewriting the control runtime, monitoring loop, environment agent, or simulator implementation.
- Introducing a framework, plugin architecture, or generalized abstraction layer.
