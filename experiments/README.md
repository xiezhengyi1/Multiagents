# Experiments

实验模块保留研究矩阵、场景资产、离线统计和统一运行入口。它不再依赖旧 orchestrator，也不自行定义 trace 格式。

## 输入与运行

每行实验输入使用 `ExperimentCase`：

- 控制实验：`method` 为 `multi_agent_control` 或 `single_agent_control`，必须提供 `user_input`。
- 环境生成：`method` 为 `environment`，必须提供 `environment_request`。

后端必须显式实现 `ExperimentBackend.prepare(case)`，返回 episode 目录和执行函数。当前没有默认后端，也不会在后端不可用时切换到模拟实现。

真实 OAI/free5GC 联动使用后端 `experiments.backends.live_testbed:create_backend`。该后端要求显式配置 testbed 接口变量、`vLLM_URL`、`vLLM_MODEL_NAME`、`VLLM_MODEL_PATH`、`VLLM_API_KEY`、`VLLM_TIMEOUT_SECONDS` 和 `MULTIAGENTS_EPISODE_ROOT`，只执行 `multi_agent_control` 用例，不会降级到模拟器或远程模型。上下文计数直接使用 `VLLM_MODEL_PATH` 中的本地 tokenizer，不访问外网。

旧版提交 `9649f526eb47c32c26db86ae214074981eea8f3e` 中属于 S2 的 30 条 canonical task 已逐字固化在 `experiments/cases/s2_original_30.json`。使用 `build_s2_cases.py` 为一次运行生成当前 `ExperimentCase` JSONL；原 task id、类别、预期对象、优化方向和成功判据均保存在 case metadata 中。

```powershell
uv run python -m experiments.scripts.build_s2_cases `
  --run-id s2-live-20260902 `
  --output experiments/results/s2-live-20260902/cases.jsonl
```

```powershell
uv run python -m experiments.scripts.run_experiment `
  --backend package.module:create_backend `
  --cases data/experiment_cases.jsonl `
  --output data/experiment_results.jsonl
```

新仿真环境尚未对接，因此 backend 由其接口确定后提供。`configs/`、`scenarios/`、`tasks/` 和离线分析脚本继续作为研究资产保留。

## 输出

runner 验证后端返回的 terminal event 与 episode journal 中的最后事件一致。原始训练数据仍只存在于对应分区的 `events.jsonl` 与 `traces.jsonl`；`ExperimentResult` 保存实验索引、episode 目录、terminal reward，以及从原生日志派生的 `EpisodeMetrics`。派生指标包括 round、repair、执行尝试、冲突、目标违例、执行失败、首轮成功和闭环恢复，不会写回 episode journal。
