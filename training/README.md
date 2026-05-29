## Training

This directory is the active home for offline trajectory tooling.

### Scope

- collect workflow trajectories
- project raw traces into training records
- export ChatML datasets
- evaluate per-agent and workflow trajectories

### Entry Scripts

- `collect_workflow_trajectories.py`
- `project_agent_trajectories.py`
- `project_workflow_trajectories.py`
- `export_agent_trajectories_to_chatml.py`
- `evaluate_workflow_trajectories.py`

### Storage Layout

- `training/<agent>/raw_traces/`
- `training/<agent>/processed/`
- `training/<agent>/datasets/`
- `training/<agent>/exports/`
- `training/<agent>/evals/`
- `training/<agent>/rejects/`

The old root-level `sft_data/` directory is legacy material. New scripts in this workspace should target `Multiagents/training`.
