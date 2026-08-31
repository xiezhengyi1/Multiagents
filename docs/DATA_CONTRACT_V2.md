# 统一数据契约 V2

本文档是新仿真环境、多代理运行时和 SFT 数据生成共用的规范。所有新数据使用 `schema_version = "2.0.0"`，不在 V2 路径中解析、修复或兼容旧 trace。

## 1. 契约边界

| 边界 | 权威类型 | 强制约束 |
| --- | --- | --- |
| 运行作用域 | `ExecutionScope` | `session_id` 和 `thread_id` 必填；仿真、轮次、SUPI 信息只能通过 scope 传递 |
| 模型输入 | `AgentInvocation` | 只含调用消息和 `TraceMetadata`；system prompt 属于 agent 契约 |
| 模型输出 | `AgentRunResult` | 保留完整原生消息顺序和经 response schema 解析的结构化输出 |
| 消息 | `AgentMessage` / `ToolCall` | 仅允许 `system/user/assistant/tool`；所有并行 tool call 必须闭合后才能继续对话 |
| 工具 | `ToolContract` / `ToolCatalog` | 执行与 trace 共用同一目录；必须声明输入 schema、副作用、结果格式和有界结果 |
| 代理间交接 | `ArtifactEnvelope` | 必须声明 payload schema、源/目标 agent、scope、correlation 和上游 artifact |
| 记忆 | `MemoryRecord` / `MemoryQuery` / `MemoryBundle` | 每次读写显式携带 scope；记忆记录携带来源 agent 和源 trace id |
| RAG | `KnowledgeQuery` / `KnowledgeResponse` | 返回结构化 hit；每个 hit 必须有 spec、release、version、anchor 引用链 |
| Trace | `AgentTrace` / `TraceTarget` | 记录 scope、prompt 版本、tool catalog 指纹、模型实际可见消息、结果来源和错误 |
| 训练投影 | `ProjectedTraceRecord` / `WorkflowTrajectoryRecord` | 身份信息只通过 `ExecutionScope` 传递，不重复维护 session/snapshot/scenario 字段 |

上述契约均拒绝未知字段，且对象创建后禁止属性重赋值。生产路径不使用“猜测字段”、多形态 JSON 提取、Markdown 代码块剔除或错误转正。

记忆命名空间有明确边界：短期 `event` 按 `thread_id` 隔离，长期 `summary` 按 SUPI 共享（无 SUPI 时按 `session_id`），摘要游标按 `thread_id` 隔离。

编排器内部的 `UnifiedControlPlan`、`ControlRoundResult` 和 round execution 入口同样只接收 `ExecutionScope`，不再平铺复制 session、snapshot、round 和 SUPI。`ControlRoundTrace`、domain proposal、agent contribution、handoff、协商和重入消息全部使用严格 Pydantic 契约；未知字段或错误 JSON 类型直接拒绝，不再通过 `dict(...)`、空对象或过滤无效列表项来“修复”输入。

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

Trace 只有同时满足以下条件才能投影为 `SftExampleV2`：

- `status=success` 且 `source=observed_llm`；
- 恰好一个可训练的 `observed_model_output`；
- 至少一个 `validated_output` 或 `compiled_output` 作为质量证据；
- 工具调用序列完整闭合，且所有工具都存在于 trace 的工具目录；
- 最后一条 assistant 消息是合法 JSON，并与观测 target 逐值相等；
- 消息中不包含 `<think>` 私有推理标签。

Qwen3 导出仅是从 `SftExampleV2` 到 Hermes/OpenAI 风格的无损投影：保留原消息顺序、原生 `tool_calls`、tool result 和 JSON Schema 工具定义，不生成伪 ChatML 标记。

## 5. 存储与版本规则

- 原始 trace 按 agent 追加写入 `training/<agent>/raw_traces/<agent>.jsonl`。
- 处理后的通用 SFT 和 Qwen3 记录写入对应 agent 的 `processed/` 目录。
- 任何 prompt、response schema 或工具集变更都必须更新 prompt/contract 版本；工具集变更会自动改变 catalog fingerprint。
- 不允许将 validator/compiler/runtime 派生值伪装成模型 label。如需训练该值，必须为它建立独立 prompt、输入和响应 schema 契约。
