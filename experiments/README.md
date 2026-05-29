# Experiments

这个目录把第 4 章实验计划落成一套统一的实验资产，包含：

- 实验矩阵配置
- 任务集与人工标注模板
- 单智能体基线目录
- 结果台账与图表输出目录
- 与当前仓库主流程对接的运行脚本

## 目录结构

- `configs/`
  - `methods.json`: `B1/B2/B3/Ours` 方法矩阵
  - `scenarios.json`: `S1/S2/S3` 及 `S1P/S2P/S3P` 场景定义
  - `experiment_matrix.json`: `E1-E5` 与 `E1P-E5P` 实验分组
- `tasks/`
  - `task_catalog.json`: 20 条控制任务，覆盖 4 类请求
  - `annotation_template.csv`: 人工标注与验收模板
- `single_agent/`
  - 单智能体基线说明与配置占位
- `results/`
  - 原始运行记录、汇总表、失败案例、案例分析、图表输出
- `scripts/`
  - `build_user_inputs.py`: 从 `task_catalog.json` 生成批量实验输入
  - `run_method.py`: 运行指定方法版本
  - `aggregate_results.py`: 把原始运行结果汇总到实验台账
  - `launch_experiments.py`: 按实验矩阵逐项启动实验

## 当前仓库能力边界

当前仓库已经具备两条实验运行链路：

- `Ours`: 多智能体闭环主流程
- `B1/B2/B3`: 单智能体推理 + 现有执行后端

场景层现在已经真正落地在：

- `Multiagents/experiments/scenarios/s1_basic_single_slice.yaml`
- `Multiagents/experiments/scenarios/s2_medium_complexity.yaml`
- `Multiagents/experiments/scenarios/s3_high_complexity.yaml`
- `Multiagents/experiments/scenarios_public/s1_basic_single_slice_public_datasets.yaml`
- `Multiagents/experiments/scenarios_public/s2_medium_complexity_public_datasets.yaml`
- `Multiagents/experiments/scenarios_public/s3_high_complexity_public_datasets.yaml`

`Multiagents/experiments/scripts/run_method.py` 会在每次方法运行前，根据 `--scenario` 重建当前图快照对应的场景初态，因此同一场景下不同方法会从一致初始态起跑。

其中 `S1P/S2P/S3P` 是公开数据集驱动的场景变体。它们保持原始对象拓扑与任务对象集合，只替换服务流的统计参数来源，用于比较人工构造画像与公开数据集代理画像对实验结论的影响。

需要直接指出的实验设计问题仍然存在：

- `S1` 依然是一个“语义解析/QoS 微型场景”，不适合承担跨切片迁移或复杂资源冲突类主实验。
- 因此 `E1` 主对比已调整为 `S2 + S3`；`S1` 只保留在 `E4` 这类轻量补充实验里。
- `E1P-E5P` 与 `E1-E5` 结构平行，但场景输入改为 `S1P/S2P/S3P`。二者不应混合汇总，否则会破坏数据来源的一致性。
- `build_user_inputs.py` 现在会严格校验任务对象是否真实存在于对应场景；一旦任务引用漂移，会直接报错而不是继续生成带病输入。

## 建议执行顺序

1. 先检查 `configs/scenarios.json` 和 `configs/experiment_matrix.json`，确认本次实验允许的场景。
2. 用 `scripts/build_user_inputs.py` 生成本次实验输入。
3. 先跑 `Ours`：

```powershell
.\.venv\Scripts\python.exe Multiagents\experiments\scripts\build_user_inputs.py --experiment E1 --scenario S2
.\.venv\Scripts\python.exe Multiagents\experiments\scripts\run_method.py --method Ours --experiment E1 --scenario S2
.\.venv\Scripts\python.exe Multiagents\experiments\scripts\run_method.py --method Ours --experiment E1 --scenario S3
```

若使用公开数据集驱动变体：

```powershell
.\.venv\Scripts\python.exe Multiagents\experiments\scripts\build_user_inputs.py --experiment E1P --scenario S2P
.\.venv\Scripts\python.exe Multiagents\experiments\scripts\run_method.py --method Ours --experiment E1P --scenario S2P
.\.venv\Scripts\python.exe Multiagents\experiments\scripts\run_method.py --method Ours --experiment E1P --scenario S3P
```

4. 运行单智能体基线：

```powershell
.\.venv\Scripts\python.exe Multiagents\experiments\scripts\run_method.py --method B1 --experiment E4 --scenario S2
.\.venv\Scripts\python.exe Multiagents\experiments\scripts\run_method.py --method B2 --experiment E4 --scenario S2
.\.venv\Scripts\python.exe Multiagents\experiments\scripts\run_method.py --method B3 --experiment E2 --scenario S2
```

5. 按配置批量启动：

```powershell
.\.venv\Scripts\python.exe Multiagents\experiments\scripts\launch_experiments.py
```

只跑某一组：

```powershell
.\.venv\Scripts\python.exe Multiagents\experiments\scripts\launch_experiments.py --experiment E1 --scenario S2
```

批量启动时，单项失败会自动跳过并继续执行，其失败记录写入：

- `results/ledgers/failed_experiments.jsonl`

如果只给 `--experiment` 不给 `--scenario`，而该实验跨多个场景，`build_user_inputs.py` 会拒绝生成含混记录。这是刻意收紧后的行为。

## 结果约定

- 每次运行都写入 `results/ledgers/run_ledger.csv`
- 原始 JSONL 放在 `results/raw_runs/`
- 汇总表放在 `results/summaries/`
- 典型失败恢复案例放在 `results/case_studies/`
