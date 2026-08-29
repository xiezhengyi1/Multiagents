## Training

This directory is the active home for offline trajectory tooling.

### Scope

- collect workflow trajectories
- project raw traces into training records
- export ChatML datasets
- evaluate per-agent and workflow trajectories

### Modules

- `schemas.py`: persisted training record models
- `artifacts.py`: training paths and JSON/JSONL I/O
- `projection.py`: runtime trace, workflow, fine-tuning, and ChatML projection
- `project.py`: unified projection CLI
- `collect_workflow_trajectories.py`: online workflow collection
- `evaluate_workflow_trajectories.py`: offline trajectory evaluation

Run entry points as modules:

```powershell
python -m training.collect_workflow_trajectories
python -m training.project agents
python -m training.project workflows
python -m training.project chatml --agent intent_encoding
python -m training.evaluate_workflow_trajectories
```

### Storage Layout

- `training/<agent>/raw_traces/`
- `training/<agent>/processed/`

The old root-level `sft_data/` directory is legacy material. New scripts in this workspace should target `Multiagents/training`.
