# Workflow Trajectory Pipeline

`sft_data/` 现在只保留一条最小主线，对应三个目标：

1. 全流程自动化收集 trajectory
2. 对每个 trajectory 评估 tool 调用正确性
3. 对整条 workflow trajectory 做整体评估
4. 将 raw run tree 投影成带 tool 的 agent/workflow trajectory，并可继续导出为 ChatML

## Kept Scripts

- `sft_data/collect_workflow_trajectories.py`
  直接消费 `generate_user_inputs.py` 产出的 `.json/.jsonl`，批量运行 `MainControlOrchestrator`，并把每次 workflow run 的结果写到 `sft_data/workflow/processed/trajectory_runs_v1.jsonl`。
- `sft_data/evaluate_workflow_trajectories.py`
  基于 session 关联各 agent 的 raw traces，输出两类评估：
  - trace 级 tool correctness
  - workflow 级整体 trajectory judgement
- `sft_data/project_agent_trajectories.py`
  把各 agent 的 raw run tree 投影成 `ProjectedTraceRecord`，保留 `message_trajectory`、`tool_calls`、`tool_results`。
- `sft_data/project_workflow_trajectories.py`
  基于 `trajectory_runs_v1.jsonl` 和同 session 的 agent trajectories 生成带 tool 的 workflow trajectory。
- `sft_data/export_agent_trajectories_to_chatml.py`
  把单个 agent 的 projected trajectory 直接导出成 ChatML JSONL。
- `sft_data/common.py`
  公共路径和文件工具。
- `sft_data/schemas.py`
  trace 投影和数据记录的共享 schema。

## Data Layout

```text
sft_data/
|-- collect_workflow_trajectories.py
|-- evaluate_workflow_trajectories.py
|-- project_agent_trajectories.py
|-- project_workflow_trajectories.py
|-- export_agent_trajectories_to_chatml.py
|-- common.py
|-- schemas.py
|-- workflow/
|   `-- processed/
|       |-- trajectory_specs_v1.jsonl
|       |-- trajectory_runs_v1.jsonl
|       |-- workflow_trajectories_v1.jsonl
|       |-- trajectory_evaluations_v1.jsonl
|       `-- trajectory_evaluation_summary_v1.json
|-- main_control/
|   `-- raw_traces/
|-- intent_encoding/
|   `-- raw_traces/
|-- policy_dispatch/
|   |-- raw_traces/
|   `-- processed/
`-- optimization_strategy/
    `-- raw_traces/
```

说明：

- `raw_traces/*.jsonl` 仍然保留为各 agent 的运行时原始 trace 存储。
- 历史 `datasets/exports/evals/processed` 数据目录不会被这个新主线主动删除，因为它们是已有实验产物，不是脚本。
- 评估脚本默认按 `trajectory_runs_v1.jsonl` 中的 `session_id` 去关联各 agent raw traces，不要求预先清空旧 trace 文件。

## Commands

```powershell
.\.venv\Scripts\python.exe sft_data\collect_workflow_trajectories.py --user-inputs generated_user_inputs.json
.\.venv\Scripts\python.exe sft_data\project_agent_trajectories.py --agent intent_encoding --agent optimization_strategy --agent policy_dispatch
.\.venv\Scripts\python.exe sft_data\project_workflow_trajectories.py
.\.venv\Scripts\python.exe sft_data\export_agent_trajectories_to_chatml.py --agent intent_encoding
.\.venv\Scripts\python.exe sft_data\evaluate_workflow_trajectories.py
```

如果不传 `--user-inputs`，采集脚本会优先读取项目根目录的 `generated_user_inputs.json`；如果文件不存在，则按当前网络快照现场生成输入。
