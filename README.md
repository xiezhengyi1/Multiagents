# Multiagents

这是当前重构后的主工作区。实验、训练、控制运行时、知识运行时都以这个目录为准，不再以外层旧目录为准。

## 当前主结构

```text
Multiagents/
  agent_runtime/
  database/
  model/
  src/
    control_runtime/
    knowledge_runtime/
    shared/
  experiments/
    configs/
    scenarios/
    tasks/
    scripts/
    results/
  training/
  knowledge_build/
  tests/
  docs/
  data/
    runtime/
```

## 控制逻辑

三智能体主链路保持为：

1. `Main`
   - 负责高层路由、轮次策略、失败归因后的重试入口选择
   - 产出 `requested_domains`、`supi`、`retry_scope`、`next_agent`、`intent_encoding_guidance`
   - 不负责 app/flow/association 的最终 grounding
2. `IEA`
   - 负责实体解析和语义 grounding
   - 必须把 app/flow/AM target 绑定到工具或缓存证据
   - 产出可复用的 `OperationIntent`
3. `OSA`
   - 负责优化预览、策略参数选择、修订轮策略生成
   - 输出必须和 optimizer preview / runtime evidence 交叉一致

诊断职责已收回到主控闭环，不再把 diagnosis 当成独立主智能体阶段。

## 运行实验

在当前目录下执行：

```bash
python experiments/scripts/build_user_inputs.py --experiment E1 --scenario S2
python experiments/scripts/launch_experiments.py --method B1 --experiment E1 --scenario S2
```

前提：

- 需要使用项目依赖环境运行，而不是裸 `C:\Python314\python.exe`
- 当前项目已在 `pyproject.toml` 中声明 `PyYAML` 等依赖；如果直接用系统 Python 运行，脚本会因为缺依赖而失败

常用方式：

```bash
# 全量实验矩阵
python experiments/scripts/launch_experiments.py

# 指定方法
python experiments/scripts/launch_experiments.py --method B1

# 指定实验与场景
python experiments/scripts/launch_experiments.py --method B1 --experiment E1 --scenario S2

# 只生成输入与执行，不聚合结果
python experiments/scripts/launch_experiments.py --method B1 --experiment E1 --scenario S2 --skip-aggregate
```

说明：

- `build_user_inputs.py` 先按 `experiment/scenario` 过滤任务，再做一致性校验。
- `launch_experiments.py` 会串联：
  - 生成输入
  - 调 `run_method.py`
  - 最后聚合结果
- `run_method.py` 继续调用 `experiments/scripts/` 下的：
  - `run_workflow_experiment.py`
  - `run_single_agent_experiment.py`

## 与外层旧目录的关系

外层目录仍有旧代码和旧脚本残留，例如：

- `experiment/`
- `sft_data/`
- 旧版根目录 `README.md`

这些内容现在只应视为迁移残留。新的改动应优先落在本目录。

## 设计原则

- 不加 fallback 来掩盖真实问题
- 不保留假的包边界、假的 compiler 契约、假的 validator 契约
- Main 强化，但不越权做 grounding
- IEA 必须证据化 grounding
- OSA 必须对 optimizer 结果负责，而不是自由生成不可执行 payload

## 进一步阅读

- [STRUCTURE.md](C:\Users\xiezhengyi\Desktop\research\6G+AI\code\Multiagents\Multiagents\STRUCTURE.md)
- [CONTROL_RUNTIME_LOGIC.md](C:\Users\xiezhengyi\Desktop\research\6G+AI\code\Multiagents\Multiagents\CONTROL_RUNTIME_LOGIC.md)
- [CONTROL_RUNTIME_MINDMAP.md](C:\Users\xiezhengyi\Desktop\research\6G+AI\code\Multiagents\Multiagents\CONTROL_RUNTIME_MINDMAP.md)
- [RESEARCH_ARCHITECTURE_THREE_AGENT.md](C:\Users\xiezhengyi\Desktop\research\6G+AI\code\Multiagents\Multiagents\RESEARCH_ARCHITECTURE_THREE_AGENT.md)
