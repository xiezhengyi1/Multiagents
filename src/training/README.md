# Training

训练模块不改变运行时记录，只从原生 episode 显式投影数据。

## 原生输入

`EpisodeRecords` 只接受恰好两个文件：

- `events.jsonl`：完整 `CoordinationEvent` 图和 terminal reward。
- `traces.jsonl`：每次真实模型请求对应一条 `AgentTrace`。

## SFT

`project_sft_examples` 只选择：

- 模型调用成功；
- 输出事件有且仅有一个 validator resolution；
- 输出被接受，或输出本身有效但触发了上游重绑定；
- 被 validator 明确 invalidated 的输出不会进入 SFT。

每条 `SftExample` 保留该次调用的真实 `input_messages`、单条 `output_message`、可见工具 schema 以及对应输出/验证事件 ID。

## RL

`project_rl_trajectory` 保留完整 events、全部原子 traces 和 terminal reward。修复边、工具观察、确定性编译与执行反馈都留在轨迹中，不会被伪装成模型输出。

```powershell
uv run python -m training.project `
  --dataset-root data/runtime `
  --partition multi_agent_control `
  --sft-output data/projected/sft.jsonl `
  --rl-output data/projected/rl.jsonl
```
