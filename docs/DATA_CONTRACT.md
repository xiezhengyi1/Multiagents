# 统一数据契约

本文档是新仿真环境、多代理运行时和 SFT 数据生成共用的唯一规范。数据中不携带 schema 版本；契约变更直接替换旧结构，不解析、修复或兼容旧 trace。

## 1. 契约边界

| 边界 | 权威类型 | 强制约束 |
| --- | --- | --- |
| 运行作用域 | `ExecutionScope` | `session_id` 和 `thread_id` 必填；仿真、轮次、SUPI 信息只能通过 scope 传递 |
| 模型输入 | `AgentInvocation` | 只含调用消息和 `TraceMetadata`；system prompt 属于 agent 契约 |
| 模型输出 | `AgentRunResult` | 保留完整原生消息顺序和经 response schema 解析的结构化输出 |
| 消息 | `AgentMessage` / `ToolCall` | 仅允许 `system/user/assistant/tool`；所有并行 tool call 必须闭合后才能继续对话 |
| 工具 | `ToolContract` / `ToolCatalog` | 执行与 trace 共用同一目录；必须声明输入 schema、副作用、结果格式和有界结果 |
| 代理间交接 | `ArtifactEnvelope` | 必须声明 request/response 方向、唯一 payload 类型、源/目标 agent、scope、correlation 和上游 artifact |
| 记忆 | `MemoryRecord` / `MemoryQuery` / `MemoryBundle` / `MemoryContextView` | 每次读写显式携带 scope；跨 Agent 上下文同时保留 query、入选记录和最终 prompt 文本 |
| RAG | `KnowledgeQuery` / `KnowledgeResponse` | 返回结构化 hit；每个 hit 必须有 spec、release、version、anchor 引用链 |
| 意图落地 | `GroundingDecision` / `GroundingEvidenceBundle` | flow、mobility、授权和证据分别建模；未知别名和未解析身份直接拒绝 |
| 策略规划 | `PolicyPlanDraft` / `OptimizerPlanEvidence` | scope 唯一承载身份；优化器证据、策略状态、缺口和问题全部显式 |
| 冲突与诊断 | `ConflictResolutionRequest/Result` / `AssuranceDiagnosisRequest/Result` | 资源视图、上游上下文、执行反馈和裁决使用唯一嵌套结构 |
| Trace | `AgentTrace` / `TraceTarget` | 记录 scope、prompt 版本、tool catalog 指纹、模型实际可见消息、结果来源和错误 |
| 训练投影 | `ProjectedTraceRecord` / `WorkflowTrajectoryRecord` | 身份信息只通过 `ExecutionScope` 传递，不重复维护 session/snapshot/scenario 字段 |

上述契约均拒绝未知字段，且对象创建后禁止属性重赋值。数据生成路径不使用“猜测字段”、多形态 JSON 提取、Markdown 代码块剔除或错误转正。

记忆命名空间有明确边界：短期 `event` 按带类型前缀的 `thread_id` 隔离，长期 `summary` 按带类型前缀的 SUPI 共享（无 SUPI 时按 `session_id`），摘要游标按 `thread_id` 隔离。`MemoryBundle` 会验证短期记录只能是同一 thread 的 event、长期记录只能来自同一 SUPI/session 命名空间的 summary；记忆 ID 和 trace 来源不得重复。`MemoryQuery.top_k` 必须由调用方显式给出。`MemoryContextView` 记录实际 query、重排后入选的短期/长期记录和模型可见文本；未执行检索时必须显式 `query=null` 且不得携带来源不明的记忆文本。

RAG 工具在执行检索前创建 `KnowledgeQuery`。非法类型或超出 `1..30` 的 limit 直接拒绝，不进行类型转换、默认值替换或上下界裁剪。`KnowledgeResponse` 的 hit 数不得超过请求 limit，同一响应中的引用 key 必须唯一，分数必须是有限浮点数。

`ArtifactEnvelope.kind` 明确区分 request 与 response；request 不得声明上游 artifact，response 必须声明至少一个上游 artifact，源和目标 agent 不得相同。payload 的唯一类型判别字段是 `artifact_type`，不再同时维护同值的 `payload_schema_name`。存储、缓存和 worker 均校验 envelope 方向，不依赖调用方法名称在契约外猜测。

编排器内部的 `UnifiedControlPlan`、`ControlRoundResult` 和 round execution 入口同样只接收 `ExecutionScope`，不再平铺复制 session、snapshot、round 和 SUPI。`ControlRoundTrace`、domain proposal、agent contribution、handoff、协商和重入消息全部使用严格 Pydantic 契约；未知字段或错误 JSON 类型直接拒绝，不再通过 `dict(...)`、空对象或过滤无效列表项来“修复”输入。

Main 模型直接以严格 `MainControlDecision` 作为 response schema，不再经过 `extra=allow` 的 wire 模型。所有模型自有顶层字段都是 JSON Schema required；未知字段、缺失字段、重复集合、无效 `current_stage` 和重复 stage index 直接拒绝。运行时不再补写 semantic target 的 SUPI、不再裁剪 current stage，也不再维护代表旧 wire 修复次数的 schema projection 指标。

编译后的 `GlobalControlIntent` 只通过 `ExecutionScope` 保存 session、thread、snapshot、round 和顶层 SUPI，不再平铺复制这些身份字段。Main compiler 只生成明确属于运行时的 routing、round strategy 和 reuse contract；retry 输出缺少 `retry_scope` 或 intent-encoding guidance 时返回验证错误，不能由 compiler 自动补文案或默认重试范围。

`ObjectiveProfile` 是 Main、Single、Planning 与 optimizer 共用的唯一目标函数契约。profile 名和五项 cost 全部必填、未知字段直接拒绝；Main 输出缺少任一权重都会失败，不能再用 balanced 默认值补齐。确定性 Single 路径只能通过显式标准 profile 工厂创建完整权重对象。

`PlanningContext` 不再复制 session、snapshot 和 round，身份信息只取自 `shared_context.main_intent.scope`；memory 字段使用 `MemoryContextView`，不再传递裸字符串。`OptimizerInput` 和 `JointOptimizationRequest` 同样只接收 `ExecutionScope`，并强制 snapshot、目标 SUPI 和控制域非空且不重复。optimizer 配置的 template、solver mode、目标、约束、决策变量和 QoS 模式全部显式构造；请求工厂不再覆盖 Main 已确定的 objective profile。可执行 SM 策略的带宽必须来自同一 flow 的合法优化器分配，缺失、非数值或非正值直接失败，不沿用模型建议值。

Planning 与 Single 共用严格 `OsaAdvisorOutput`。根字段（包括 `open_questions`）以及 SM、AM、URSP 的所有嵌套字段全部是 JSON Schema required；可空字段也必须显式返回 null。`partial_policies` 只有 snake_case 的单一结构，不解析 camelCase 别名、不接受顶层 `blocked_reason`，SUPI 和 policy_id 由 compiler 生成。Planning/Single prompt 版本为 `2.2.0`；Grounding prompt 与 trace metadata 统一为 `2.1.0`。

`GroundingDecision` 使用唯一的 `FlowSelector`、`MobilityIntent`、`GroundingEvidenceBundle`、`QosOperationConstraint` 和 `SliceMigrationAuthorization` 结构。resolved flow 必须同时具有 SUPI、app_id 和 flow_id；unresolved flow 不能声明 flow_id。mobility 只接受 association、RFSP、allowed/target S-NSSAI、access type 和 service-area restriction 的规范字段，不读取旧别名。

`PolicyPlanDraft`、`ConflictResolutionRequest`、`AssuranceDiagnosisRequest` 和 `UnifiedControlPlan` 的身份均只来自 `ExecutionScope`。Mediator 的资源视图、冲突范围与 upstream context，Diagnosis 的 execution feedback 与 upstream context，以及所有结果集合都必须显式提供；不再通过空构造器或扁平 session/snapshot 字段补齐。

## 2. 模型工具协议

工具定义的唯一公开形式是 JSON Schema；所有工具必须声明：

- `name` 和非空 `description`；
- `input_schema.type = "object"`；
- `side_effect` 为 `read`、`write` 或 `external_action`；
- `result_format` 为 `json` 或 `text`；
- 可选的 `output_schema` 和 capability 列表。

模型调用工具时使用原生 `assistant.tool_calls[]`，工具结果使用 `role=tool` 且必须回填同一 `tool_call_id`。未知工具、非对象参数、重复 call id、孤儿结果、未闭合调用或工具执行失败都直接使该次调用失败。执行循环直接消费 trace 中同一个 `ToolCatalog`；`json` 工具必须返回合法 JSON，`text` 工具必须返回字符串，工具本身不得越过边界创建协议级 `ToolMessage`。

工具输出不会被静默截断，也不会在后续轮次替换成伪造的“已压缩”消息。超过 `ContextPolicy` 字符上限时调用明确失败，由工具端按其输出契约返回有界结果。因此 trace 保存的消息与模型实际看到的消息逐条一致。

## 3. Trace 语义

一次完成的模型调用对应一条 `AgentTrace`。如果业务验证在模型返回后发生，调用方延迟写 trace，并且只能二选一：

1. 验证成功：写 `status=success`，附加 validator/compiler target；
2. 验证失败：写 `status=error`，保留原始模型消息和明确错误，不伪造已验证 target。

每个重试都是新的模型调用，因此每个已完成的重试结果都单独写 trace。调用阶段直接抛错由统一 trace wrapper 写入；调用后验证结果由 agent 显式写入，不允许两条路径重复记录。

`TraceTarget` 的来源是受限的：

| kind | producer | 含义 | 可直接用作 SFT label |
| --- | --- | --- | --- |
| `observed_model_output` | `model` | 模型实际输出 | 是 |
| `validated_output` | `validator` | 业务验证后的派生结果 | 否 |
| `compiled_output` | `compiler` | 编译/组装后的派生结果 | 否 |
| `deterministic_output` | `runtime` | 确定性运行时结果 | 否 |

SFT 资格是由 `kind == observed_model_output` 唯一推导的只读属性，不是可写入或可序列化的状态字段，避免 target 类型与资格标记发生漂移。

确定性执行 trace 与模型 trace 共用容器契约，但 `source=deterministic_runtime`，不能声称观测到模型输出。

## 4. SFT 质量门

Trace 只有同时满足以下条件才能投影为 `SftExample`：

- `status=success` 且 `source=observed_llm`；
- 恰好一个可训练的 `observed_model_output`；
- 至少一个 `validated_output` 或 `compiled_output` 作为质量证据；
- 工具调用序列完整闭合，且所有工具都存在于 trace 的工具目录；
- 最后一条 assistant 消息是合法 JSON，并与观测 target 逐值相等；
- 消息中不包含 `<think>` 私有推理标签。

Qwen3 导出仅是从 `SftExample` 到 Hermes/OpenAI 风格的无损投影：保留原消息顺序、原生 `tool_calls`、tool result 和 JSON Schema 工具定义，不生成伪 ChatML 标记。

## 5. 存储规则

- 原始 trace 按 agent 追加写入 `training/<agent>/raw_traces/<agent>.jsonl`。
- 处理后的通用 SFT 和 Qwen3 记录写入对应 agent 的 `processed/` 目录。
- prompt 变更时更新 `prompt_version`；工具集变更会自动改变 catalog fingerprint。数据契约本身不携带版本字段。
- 不允许将 validator/compiler/runtime 派生值伪装成模型 label。如需训练该值，必须为它建立独立 prompt、输入和响应 schema 契约。
