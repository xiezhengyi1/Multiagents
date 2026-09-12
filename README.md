# Multiagents

面向 6G 策略控制研究的数据生成运行时。当前重构的目标不是建设通用平台，而是让多智能体协同、单智能体基线和环境生成使用同一套可追溯数据契约，直接服务 SFT 与 RL。

## 核心链路

多智能体控制图为：

```text
environment snapshot
  -> Main -> validator
  -> SBA  -> validator -> deterministic compiler
  -> PPA  -> validator -> typed policy compiler
  -> policy mediator -> execution -> assurance diagnoser
  -> episode controller -> terminal / repair reentry
```

修复不是固定单向链：PPA 可以请求 SBA 重新绑定证据，SBA 可以请求 Main 重新路由，超过预算后交给 episode controller 终止。所有正向交接、验证、修复、观察、执行反馈和终止都记录为 `CoordinationEvent`。

三个模型 actor 保留原 Main、IEA/SBA、OSA/PPA prompt，只替换其中的数据契约说明。模型只生成 `CoordinationDecision`；事件编号、来源、作用域和引用关系由运行时写入。

PPA 的模型提案不会直接作为 PCF 请求发送。确定性编译器将其转换为 `src/model` 中的 3GPP 类型：

- `SmPolicyDecision`
- `PcfAmPolicyControlPolicyAssociation`
- `UrspRuleRequest`

因此同时保留模型原始输出和可执行类型化产物，训练时可分别选择。

## 目录

```text
src/
  control_runtime/
    actors/        # Main、SBA、PPA、single、validator、诊断与controller
    graph/         # LangGraph 路由、事件上下文、修复预算
    simulation/    # 新仿真环境必须实现的统一边界与观察工具
    execution/     # 类型化策略编译、提交、回执校验、指标评估
    monitoring/    # 快照指标检测和 episode-local 生命周期
    integrations/  # 暂停迁移，等待新仿真环境接口确定
  environment/     # 独立的环境生成 workflow
  knowledge_runtime/ # 可拆装 RAG 工具和检索后端
  model/           # 3GPP 输出模型
  shared/          # 唯一通信、trace、journal 和 agent 工厂
  training/        # 原生 episode 读取及显式 SFT/RL 投影
experiments/       # 实验契约、runner、矩阵与离线分析
knowledge_build/   # 知识库构建和评估
database/          # 知识库持久化
tests/
docs/
```

## 数据集

运行时只落地：

```text
<dataset_root>/<partition>/<episode_id>/
  events.jsonl
  traces.jsonl
```

分区固定为：

- `multi_agent_control`
- `single_agent_control`
- `environment`

SFT 与 RL 文件是离线投影，不是运行时的第二套真相：

```powershell
uv run python -m training.project `
  --dataset-root data/runtime `
  --partition multi_agent_control `
  --sft-output data/projected/multi_agent_sft.jsonl `
  --rl-output data/projected/multi_agent_rl.jsonl
```

## 当前边界

- OAI/free5GC 仿真环境已通过 `TestbedSimulatorAdapter` 提供 snapshot、read 和 execute 能力。
- 旧 `agents/diagnostics/orchestrators` 已完成行为迁移并删除。
- `integrations` 以仿真平台的原生 snapshot、action 和 free5GC 订阅数据为唯一事实源，不再读取旧数据库、NS-3 策略端点或随机拓扑。
- 完整联合优化直接使用 snapshot 中的 slice 双向带宽容量与负载、gNB CPU/PRB 容量与利用率、UPF CPU/内存容量与利用率。
- RAG 是显式注入的 LangChain 工具，不向共享上下文偷偷注入内容。
- 长期记忆暂缓；当前只保留 episode 事件祖先上下文。
- 不接受 `schema_version`，不兼容或修复旧 trace。
- 不使用异常兜底来切换模型、检索器或执行后端。

本地运行前必须显式配置 `TESTBED_SNAPSHOT_BASE_URL`、`TESTBED_ACTION_BASE_URL`、`TESTBED_SUBSCRIBER_BASE_URL`、`TESTBED_RUN_ID`、`TESTBED_PLMN_ID`、`TESTBED_ACTION_API_KEY` 和 `TESTBED_HTTP_TIMEOUT_SECONDS`。入口为 `control_runtime.integrations.create_live_testbed_adapter()`。

详细契约见 [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md)，架构说明见 [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md)。
