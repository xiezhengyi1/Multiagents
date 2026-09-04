# 技术架构

## 运行时真相

系统只有两个原生日志：协调事件与原子模型调用。LangGraph 负责执行图，不定义另一套持久化 state；LangChain 负责模型与工具调用，不定义另一套业务消息。

```text
LangGraph state(current_event)
  -> actor
  -> CoordinationDecision
  -> EpisodeJournal
  -> CoordinationEvent
  -> next actor
```

模型 actor 的 `AgentTrace` 由调用包装层记录，和事件通过 `trigger_event_id` 及模型输出事件的 `in_reply_to` 建立关系。

## 控制工作流

`ControlWorkflow` 注册 Main、SBA、PPA、三个 validator、两个 deterministic compiler、policy mediator、execution dispatcher、assurance diagnoser 与 episode controller。模型 proposal/repair_response 会先被 router 拦截到对应 validator；validator 决定接受、本地修复或向上游修复。partial plan、策略冲突、执行失败和目标违例都通过同一修复预算重新进入责任 actor。

`SingleControlWorkflow` 只有一次模型 actor，但使用同一观察工具集合、PPA proposal、类型化 compiler、执行服务和 terminal reward 结构，因此它是可比较基线，不是另一套数据格式。

`EnvironmentWorkflow` 独立生成和验证仿真场景，写入 `environment` 分区。环境模型不会混入控制模型数据。

## 执行层

`control_runtime.execution.policy_compiler` 把 PPA 语义提案转换为 `src/model` 类型化 payload。`PolicyMediator` 在提交前检查重复资源目标、S-NSSAI 对齐和 UE-AMBR/SM QoS 一致性。`PolicyBatchExecutor` 只负责向注入的 `SimulatorAdapter` 提交 action 并校验 receipts；`ExecutionDispatcher` 只负责图事件转换。`AssuranceDiagnoser` 综合 receipts 与显式 objective metrics 形成结构化诊断。

## 监控层

监控从规范化 `EnvironmentSnapshot.metrics` 读取显式规则，产生 `MonitorResult` observation event。生命周期状态仅在 episode 内维护，不构成长时记忆或隐藏 prompt 上下文。

## 知识层

`knowledge_runtime` 对 agent 公开两个标准工具：语义检索与精确 key 查询。具体检索实现通过 `RagBackend` 注入。工具输出是 `KnowledgeResponse`，每个 hit 携带规范引用信息。运行时读取 `knowledge_build` 的发布元数据和构建产物，但不再导入 `knowledge_build/scripts`。

## 旧实现处理

旧 `control_runtime/agents`、`control_runtime/diagnostics` 和 `control_runtime/orchestrators` 已在行为迁移后删除。模型语义由 `actors` 保留，冲突协调与 assurance diagnosis 已拆成事件图节点，round/retry/闭环恢复指标改为日志投影，不保留旧 runtime、artifact envelope 或 orchestration result 契约。

## 暂停项

`control_runtime.integrations` 已恢复旧源码，但其 PCF、optimizer、scenario、storage 适配仍依赖旧仿真/数据库结构。根据当前决定，这部分等待新仿真环境接口后再迁移，不接入现有 workflow，也不以兼容层掩盖接口缺失。
