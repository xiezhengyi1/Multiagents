# 统一数据契约

本文件是多智能体控制、单智能体基线、环境生成、SFT 与 RL 的当前数据规范。契约不带 `schema_version`；旧结构直接失效，不做兼容解析。

## 1. 唯一通信记录

所有 actor 间通信使用 `CoordinationEvent`：

```text
CoordinationEvent
├── event_id                 # journal 生成的稳定引用键
├── event_index
├── scope
│   ├── episode_id
│   └── round_index
├── source_agent
├── target_agent
├── message_type
├── in_reply_to
├── input_artifact_ids
├── invalidated_artifact_ids
└── payload
```

`event_id` 是落盘图的引用键，不由模型生成。`event_index` 在 episode 内从 0 连续递增。所有引用只能指向更早事件；输入和失效集合不得重叠。

消息类型只有：`proposal`、`handoff`、`validation_result`、`repair_request`、`repair_response`、`observation`、`execution_feedback`、`diagnosis`、`terminal`。payload 类型声明自己允许出现在哪些消息中，类型与消息不匹配时直接拒绝。

模型输出使用更小的 `CoordinationDecision`：

```text
CoordinationDecision
├── target_agent
├── message_type
└── payload
```

运行时把 decision 扩展为 event。模型不输出 event identity、scope、source 或 artifact lineage。

## 2. 图与修复语义

正常路径是 Main → SBA → PPA → execution，但通信是有向图：

- 本地结构错误：validator 向同一模型 actor 发 `repair_request`。
- PPA 缺证据或绑定错误：PPA validator 向 SBA 请求重新 grounding。
- SBA 无法完成绑定：SBA validator 向 Main 请求重新路由或澄清。
- partial plan 根据阻塞责任重新进入 SBA 或 PPA，不直接终止。
- 执行前协调发现跨域冲突时向 PPA 发修复请求。
- 执行失败或目标未达成时，结构化 assurance diagnosis 向责任 actor 发修复请求。
- 修复预算耗尽：交给 episode controller 产生 terminal failure。

被否定的模型输出事件必须出现在 repair event 的 `invalidated_artifact_ids`。如果模型输出本身有效，只是揭示上游证据不足，则不失效该输出；训练投影将其标记为 `accepted_upstream_repair`。

## 3. 原子模型 Trace

一次真实模型请求对应一条 `AgentTrace`：

```text
AgentTrace
├── trace_id
├── trigger_event_id
├── call_index
├── scope
├── agent_name
├── model_name
├── started_at / ended_at
├── input_messages[]
├── output_message
├── tools[]
├── status
├── error
└── usage
```

工具循环、结构化输出重试或修复重试只要再次调用模型，就产生新的 trace。Trace 保存模型实际可见的消息和工具 schema；workflow 投影、编译产物和 validator 结果不写进 `output_message`。

成功 trace 必须有一条 assistant output 且没有 error；失败 trace 必须保留 error。工具调用只能引用该 trace 中已声明的工具。

## 4. 两级策略产物

第一层是模型生成的 `PpaProposal`，用于保留模型真实学习目标。第二层由确定性编译器生成 `ExecutionAction`，其 payload 使用 `src/model` 的 3GPP 类型：

- SM：`SmPolicyDecision`
- AM：`PcfAmPolicyControlPolicyAssociation`
- URSP：`UrspRuleRequest`

`ExecutionOperation.resource_type` 必须和 payload 运行时类型一致。create/update 必须有 payload；delete 不得有 payload。执行后端必须按原 operation 顺序返回同 ID 的 receipt。

编译后的 action 先经过 `policy_mediator`。其 `ConflictAssessment` 仅记录冲突、受影响 operation 和修订指令，策略正文通过 artifact lineage 引用，不再复制。执行反馈随后由 `assurance_diagnoser` 生成 `AssuranceDiagnosis`；诊断包含根因类别、受影响 operation/domain、建议动作和 reward components。

## 5. 原生落盘

每个完成 episode 只允许：

```text
<dataset_root>/<partition>/<episode_id>/events.jsonl
<dataset_root>/<partition>/<episode_id>/traces.jsonl
```

partition 为 `multi_agent_control`、`single_agent_control` 或 `environment`。一个 episode 必须且只能以一个 terminal event 结束；`reward_total` 必须等于所有 `reward_components` 之和。

## 6. SFT 投影

SFT 单元是模型单次输入输出，不是 workflow：

```text
SftExample
├── example_id              # trace_id
├── episode_id
├── actor
├── input_messages
├── output_message
├── tools
├── output_event_id
├── validation_event_id
└── disposition
```

只有 validator 接受的模型输出，以及有效但请求上游修复的输出可入选。被 invalidated、没有成功原子 trace、没有唯一 validator resolution 的输出直接拒绝。

## 7. RL 投影

`RlTrajectory` 保留一个 episode 的完整 `events`、全部 `traces`、terminal payload 和派生 `EpisodeMetrics`。reward 属于 terminal event；不在 trace 或中间 artifact 上复制维护。`EpisodeMetrics` 从事件计算 round、repair、执行尝试、冲突、目标违例、执行失败和闭环恢复情况，不构成新的运行时事实源。这样可在后续训练中自由选择 event-level、turn-level 或 episode-level credit assignment，而不修改原始数据。

## 8. 上下文、RAG 与记忆

- 模型上下文只从当前事件的祖先图构建；当前阶段不做自动摘要或长期记忆注入。
- RAG 以显式 LangChain 工具注入，查询和结果进入真实 tool messages。
- 检索后端必须显式选择 `colbert` 或 `pgvector_bge`；后端失败直接暴露，不自动切换。
- `get_by_key` 只接受精确 key，不退化为模糊搜索。

## 9. 仿真边界

OAI/free5GC 仿真环境通过 `TestbedSimulatorAdapter` 实现 `SimulatorAdapter`：

- `capture_snapshot() -> EnvironmentSnapshot`
- `observe(ObservationRequest) -> EnvironmentObservation`
- `execute(ExecutionAction) -> ExecutionFeedback`

`NativeSnapshot` 严格校验每个 flow 的双向 SLA、双向已分配带宽、priority、五元组、packet size 和 arrival rate。`storage`、`scenario` 和 `pcf` 均直接投影该 snapshot 与 free5GC 订阅数据，不维护第二份运行时事实。

slice 双向带宽容量由场景配置显式声明，gNB PRB 容量来自 NR 带宽 RB 配置，gNB/UPF CPU 与内存容量来自中央资源计划。当前负载和利用率由 flow allocation、OAI RAN 遥测和 cAdvisor 聚合得到；联合优化不从当前流量反推容量，也不伪造优化后遥测。
