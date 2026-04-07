# Training Data Layout

This directory stores dataset builders, evaluation scripts, and export utilities.

## Layout

代码按任务模块放在 `sft_data/*/*.py`，数据产物统一按 agent 落到各自目录：

```text
sft_data/
|-- intent_encoding/
|   |-- raw_traces/
|   |-- datasets/
|   |   |-- supervised/
|   |   |   |-- success/
|   |   |   `-- failure/
|   |   `-- chatml/
|   |       |-- success/
|   |       `-- failure/
|   |-- evals/
|   |   |-- semantic_judge/
|   |   `-- evalscope/
|   |       |-- dataset/
|   |       `-- runs/
|   `-- exports/
|       `-- llama_factory/
|-- optimization_strategy/
|   |-- raw_traces/
|   |-- datasets/
|   |-- evals/
|   `-- exports/
|-- rl/
|-- tool_call/
|-- build_all_training_data.py
|-- common.py
|-- formatting.py
`-- schemas.py
```

说明：

- `sft_data/tool_call/` 现在主要承载构建脚本，不再作为共享输出目录。
- EvalScope 官方 benchmark 的 dataset、run result、workdir 都应落到 `sft_data/<agent>/evals/evalscope/`。
- ChatML 微调数据显式拆成 `success` 和 `failure` 两类，便于训练与分析分离。
- `processed/` 和 `rejects/` 目录保留兼容，但新脚本默认走 `datasets/`、`evals/`、`exports/`。

## Trace Rules

Minimal raw traces keep only the fields needed to rebuild training examples:

- `trace_id`
- `timestamp`
- `agent_name`
- `session_id`
- `snapshot_id`
- `thread_id`
- `model_name`
- `input_messages`
- `message_trajectory`
- `tool_calls`
- `tool_results`
- `structured_response`
- `status`
- `error`

IEA ChatML traces additionally follow these rendering rules:

- `message_trajectory[0]` must be `role=system` and contain the exact agent system prompt.
- Final ChatML export folds `think_tool(message=...)` into an assistant turn rendered as `<think>...</think>` immediately before the corresponding real tool call.
- `think_tool` itself must not remain as a normal `<tool_call ...>` or `<tool_result>...` turn in the exported ChatML sample.
- Tool turns render as `<tool_result>RAW_TOOL_OUTPUT</tool_result>` only.
- Tool turn content must not embed `tool_call_id`, `name`, or `status` metadata in the body.
- The final assistant turn must be the raw `OperationIntent` JSON string.
- `<think>...</think>` is recorded only when the model explicitly emits it before a real tool call.

The trace writer does not store old debug-heavy fields such as `system_prompt`, `tool_specs`, `output_messages`, `latency_ms`, or raw runtime context blobs.

## Outputs

- `intent_encoding/datasets/supervised/success/iea_sft_v1.jsonl`: request/response supervised pairs.
- `intent_encoding/datasets/chatml/success/iea_chatml_sft_v1.jsonl`: 可直接用于多步 SFT 的 ChatML 成功样本。
- `intent_encoding/datasets/chatml/failure/iea_chatml_sft_v1.jsonl`: 工具使用失败但可复盘的 ChatML 样本。
- `intent_encoding/evals/semantic_judge/semantic_judge_eval_summary_v1.json`: 真实 trace 语义 Judge 汇总。
- `intent_encoding/evals/evalscope/dataset/default.jsonl` 或 `evalscope_call_decision_records_v1.jsonl`: EvalScope general_fc 本地数据集。
- `intent_encoding/evals/evalscope/runs/<model>/summary_v1.json`: EvalScope 官方指标汇总。
- `intent_encoding/exports/llama_factory/chatml/chatml_success_v1.json`: Llama-Factory ChatML 成功训练集。
- `intent_encoding/exports/llama_factory/chatml/chatml_failure_v1.json`: Llama-Factory 失败样本分析集。
- `intent_encoding/exports/llama_factory/tool_call/tool_call_decision_v1.json`: 单步工具决策 warmup 集。

## Recommended Usage

推荐把数据链路拆成三个阶段：

1. `tool_call_decision` 用于单步工具决策 warmup，先把 call / not-call 边界学稳。
2. `chatml success` 用于多步 agent 轨迹 SFT，学习 think / tool_call / tool_result / final JSON 的完整序列。
3. `chatml failure` 默认不直接并入 SFT，而是用于误差分析、Judge 回归和后续 preference 数据构造。

## Build Commands

```powershell
.\.venv\Scripts\python.exe sft_data\intent_encoding\build_iea_sft_dataset.py
.\.venv\Scripts\python.exe sft_data\intent_encoding\build_iea_chatml_sft_dataset.py
.\.venv\Scripts\python.exe sft_data\intent_encoding\evaluate_semantic_judge.py
.\.venv\Scripts\python.exe sft_data\tool_call\build_evalscope_call_decision_dataset.py --agent intent_encoding
.\.venv\Scripts\python.exe sft_data\tool_call\evaluate_evalscope_call_decision.py --agent intent_encoding
.\.venv\Scripts\python.exe sft_data\intent_encoding\export_llama_factory_datasets.py --agent intent_encoding
.\.venv\Scripts\python.exe sft_data\optimization_strategy\build_osa_sft_dataset.py
.\.venv\Scripts\python.exe sft_data\rl\build_rl_trace_dataset.py
```
