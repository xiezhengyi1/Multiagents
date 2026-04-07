# Project Architecture Overview

## Project Purpose

这个仓库的核心目标不是单纯“调用几个 Agent”，而是构建一个面向 5G/PCF 策略控制场景的闭环多智能体系统。系统接收运维或策略调整意图，结合 UE 上下文、业务流目录、网络快照和标准知识库，生成可执行的 PCF 策略，并在执行后做冲突分析、SLA 校验、诊断和记忆沉淀。

从代码实现看，项目有两条主线：

- 在线闭环主线：围绕 `system_coordinator.py` 组织的意图解析、策略生成、策略下发、校验与反馈回路。
- 离线知识主线：围绕 `knowledge_scripts/`、`database/langchain_pg.py` 和 `tools/knowledge_tool.py` 构建的 Release-18 PCF/URSP 标准知识库，用于增强意图理解和策略规划。

README 当前更偏向介绍知识库脚本，而不是整个运行时系统；如果把仓库整体理解成“知识库项目”，会低估其真正的闭环控制定位。

## Core Capabilities

项目当前具备以下核心能力：

- 将自然语言策略请求解析为结构化 `OperationIntent`，并绑定到具体 SUPI、App、Flow。
- 基于实时网络状态和优化器结果生成 `PolicyPlanDraft`。
- 将结构化策略编译、校验后下发到 PCF。
- 基于绑定的网络快照做 SLA/保障校验，而不是只看最新瞬时状态。
- 对候选策略做冲突检测，对执行失败做归因诊断。
- 将 session、artifact、handoff、stage result、episodic memory 持久化，支持可追踪闭环。
- 构建并查询面向 PCF/URSP 标准对象的知识库，供 IEA 等 Agent 做语义补充。

## Layered Structure

### 1. Orchestration Layer

- `system_coordinator.py`

这是系统总控入口。`MultiAgentSystem` 负责：

- 初始化 IEA、OSA、PDA、CR、AD 和 `MemoryManager`
- 创建 session
- 绑定当前 snapshot
- 组装跨轮 feedback context 和 memory context
- 调度多轮闭环执行
- 记录 handoff、stage result、episodic experience

这一层定义了系统真正的控制回路。

### 2. Agent Execution Layer

- `agents/intent_encoding/agent.py`
- `agents/optimization_strategy/agent.py`
- `agents/policy_dispatch/agent.py`
- `agents/conflict_resolution/agent.py`
- `agents/assurance_diagnosis/agent.py`
- `agents/BaseAgent.py`
- `agents/worker.py`

这层是具体智能体执行层。

- `BaseAgent` 统一封装 LLM、LangChain `create_agent`、结构化输出和运行时上下文。
- `ArtifactWorkerMixin` 统一封装 artifact 消费、响应写回、队列认领和缓存。
- IEA 和 OSA 是“LLM + 工具 + 结构化输出”型 Agent。
- PDA、CR、AD 更偏执行器/分析器，结构更接近确定性业务组件。

### 3. Domain Contract Layer

- `domain/policy_plan.py`
- `domain/collaboration.py`
- `domain/policy_compiler.py`
- `domain/policy_guard.py`

这层定义系统的核心数据契约。

- `OperationIntent` 表示意图解析结果。
- `PlanningRequest` 把意图和协作上下文一起交给 OSA。
- `PolicyDraft` / `PolicyPlanDraft` 表示待执行策略草案。
- `PolicyCompiler` 把规划结果整理为可执行计划。
- `PolicyGuard` 对策略结构、字段一致性和模型合法性做严格校验。

这一层使各 Agent 之间不是靠松散字典交互，而是围绕稳定 schema 协作。

### 4. Runtime and Artifact Layer

- `agent_runtime/artifacts.py`
- `agent_runtime/context.py`
- `agent_runtime/workspace.py`
- `agent_runtime/queue.py`
- `workflows/runtime_registry.py`

这层负责把 Agent 协作落到可持久化、可隔离的运行时机制上。

- `ArtifactEnvelope` 是跨 Agent 传递的统一载体，包含 `artifact_id`、`correlation_id`、`session_id`、`snapshot_id` 和 payload。
- `ArtifactStore` 按 `source_agent__target_agent` 目录对请求/响应做原子写入。
- `AgentWorkspace` 为每个 Agent 提供隔离的 `cache/`、`work/`、`logs/`。
- `FileTaskQueue` 用 lease 机制认领待处理 artifact，避免并发重复消费。
- `RuntimeAgentRegistry` 负责 worker 注册和线程池调度。

### 5. Persistence and State Layer

- `tools/db_tool.py`
- `tools/runtime_store.py`
- `tools/network_graph.py`
- `database/langchain_pg.py`

这层管理运行时状态、网络快照和知识库存储。

- `db_tool.py` 负责 session context、UE context、snapshot 读写。
- `runtime_store.py` 负责 artifact 元数据、handoff、stage result、episodic experience 等审计数据。
- `network_graph.py` 提供图结构快照，用于更稳定地表达 UE、App、Flow、Slice、Node 关系。
- `langchain_pg.py` 提供 PGVector 连接与 embedding 适配，支撑标准知识库的 dense retrieval。

### 6. Knowledge and Tool Layer

- `tools/knowledge_tool.py`
- `tools/pcf_tools.py`
- `tools/network_status.py`
- `tools/optimizer/`
- `knowledge_scripts/`

这层向 Agent 暴露外部能力。

- `knowledge_tool.py` 提供 exact + glossary + cross-spec + vector 混合检索。
- `pcf_tools.py` 负责 UE 上下文读取和向 PCF 下发策略。
- `network_status.py` 和 `tools/optimizer/` 提供网络状态摘要与优化求解能力。
- `knowledge_scripts/` 负责把标准文档加工成可被检索消费的知识库。

## Main Workflow

## 1. 用户输入进入 Coordinator

入口在 [system_coordinator.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/system_coordinator.py)。

`MultiAgentSystem.run()` 的第一步是：

- 校验输入
- 创建 `session_context`
- 绑定 `MemoryManager` 的 thread
- 获取最新网络快照元数据

这里的关键设计是：每轮规划都绑定一个明确的 `snapshot_id`，后续 OSA、PDA、AD 都围绕这个 snapshot 工作，而不是依赖漂移中的“最新状态”。

## 2. IEA 解析意图并做实体解析

IEA 在 [agents/intent_encoding/agent.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agents/intent_encoding/agent.py)。

它做两件事：

- 用 LLM + 工具把自然语言转成 `OperationIntent`
- 用本地 catalog 把模糊 flow/app 描述解析成具体 `app_id`、`flow_id`

IEA 可调用的关键工具有：

- `get_ue_context`
- `get_ue_flow_catalog`
- `search_semantic_knowledge`
- `get_knowledge_by_key`

解析后会进入 `_postprocess_operation_intent()` 和 `_resolve_operation_intent_against_catalog()`：

- 推断 SUPI
- 规范化 operation type
- 把 flow 名称解析为实际目录中的 flow
- 在唯一命中时补齐业务流 SLA 参数和 five tuple
- 在多候选时返回 `ambiguous`
- 在无命中时返回 `unmatched`

这说明 IEA 不是单纯抽象语义解析，而是“语义解析 + 运行态实体绑定”的组合。

## 3. OSA 基于意图和协作上下文生成策略

OSA 在 [agents/optimization_strategy/agent.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agents/optimization_strategy/agent.py)。

Coordinator 会把以下内容封装成 `PlanningRequest` 交给 OSA：

- `operation_intent`
- `round_index`
- `session_id`
- `snapshot_id`
- `snapshot_metadata`
- `memory_context`
- `feedback_context`
- `handoff_history`

OSA 的工作流是：

1. 读取结构化意图和协作上下文
2. 调 `fetch_network_status`
3. 调 `run_optimization_solver`
4. 让 LLM 输出结构化 `PolicyPlanDraft`
5. 对草案做归一化，补齐标准化 policy id、flow id、session/snapshot 绑定

这意味着 OSA 不是直接手写策略，而是把“网络态势 + 优化结果 + 业务意图”联合转成标准策略草案。

## 4. PDA 编译、校验、下发并提交执行结果

PDA 在 [agents/policy_dispatch/agent.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agents/policy_dispatch/agent.py)，实际执行核心在 [workflows/execution_controller.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/workflows/execution_controller.py)。

PDA 的流程是：

1. `PolicyCompiler.compile_plan()` 将 `PolicyPlanDraft` 规整为执行计划
2. `PolicyGuard.validate_policy()` 做严格校验
3. 逐条调用 `dispatch_policy_to_pcf_request`
4. 检查 ack 是否完整
5. 用 `AssuranceEvaluator.evaluate()` 做基于 snapshot 的 SLA 校验
6. 成功后把策略合并回 UE context

这条链路有几个重要特点：

- 策略必须满足严格 schema 和命名规范，不允许“差不多能跑”
- `SmPolicyDecision` 和 `UrspRuleRequest` 分别走不同约束
- flow 级策略必须带 `flow_id`
- PDA 成功后会真正把 policy 持久化回 UE 上下文，而不是只返回建议

## 5. CR 和 PDA 并行运行

Coordinator 会并行触发：

- `policy_dispatch`
- `conflict_resolution`

CR 在 [agents/conflict_resolution/agent.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agents/conflict_resolution/agent.py)。

它从两个维度检查冲突：

- binding 冲突：同一个 `supi/app_id/flow_id/target_type`
- resource 冲突：相同资源键，例如 DNN、S-NSSAI 等

CR 输出会被记录进 round trace 和 handoff 记录。

需要明确的一点是：当前实现里，CR 是并行分析和记录组件，不会直接阻断 PDA 执行。也就是说，即使 CR 返回 `unresolved`，只要 PDA 路径成功，Coordinator 仍可能给出成功结果。这是当前架构中的一个重要行为特征。

## 6. AD 在执行后做故障归因

AD 在 [agents/assurance_diagnosis/agent.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agents/assurance_diagnosis/agent.py)。

它消费：

- PDA feedback
- dispatch receipts
- assurance verdicts
- telemetry snapshot 标识
- CR 上下文

然后把问题归为：

- `execution_failure`
- `sla_violation`
- `assurance_evaluation_failure`
- `missing_evidence`
- `inconclusive`

AD 的定位不是重新规划，而是把失败证据结构化，便于下一轮闭环或后续人工排查。

## 7. 失败反馈回流到下一轮 IEA

如果 PDA 失败，Coordinator 会把 `FeedbackReport` 作为 handoff 回传给 IEA，并把反馈串成 `feedback_context`。

下一轮 IEA 收到的不只是原始用户请求，还会看到：

- short-term memory
- long-term memory summary
- 前轮 PDA 的失败原因和修正建议

这使得系统形成真正的闭环，而不是一次性串行流水线。

## Key Interfaces and Data Structures

### OperationIntent / FlowSelector

定义在 [policy_plan.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/domain/policy_plan.py)。

这是 IEA 输出给 OSA 的核心对象，包含：

- `supi`
- `app_id` / `app_name`
- `operation_type`
- `urgency`
- `flows`
- `resolution_status`

`FlowSelector` 还包含带宽、时延、丢包、抖动、priority、five tuple 和当前带宽等运行态字段。

### PlanningRequest / PlanningContext

定义在 [collaboration.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/domain/collaboration.py)。

它把单纯“用户意图”提升为“协作输入”，引入：

- round index
- snapshot metadata
- memory context
- feedback context
- handoff history

这意味着 OSA 的输入是上下文化的，不是孤立请求。

### PolicyDraft / PolicyPlanDraft / PolicyPlan

这些对象串起“规划态”和“执行态”：

- `PolicyDraft`：单条策略草案
- `PolicyPlanDraft`：OSA 输出的整包草案
- `PolicyPlan`：PDA 内部使用的执行计划

### ArtifactEnvelope

定义在 [artifacts.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agent_runtime/artifacts.py)。

它是跨 Agent 协作的基础接口，包含：

- `artifact_id`
- `artifact_type`
- `source_agent`
- `target_agent`
- `session_id`
- `snapshot_id`
- `correlation_id`
- `payload`
- `upstream_artifact_ids`

这里的设计重点是：session、snapshot 和 artifact 全部显式建模，链路可追踪。

### AgentRuntimeContext

定义在 [context.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agent_runtime/context.py)。

这是注入给 LangChain Agent 的运行时上下文，包含：

- `agent_name`
- `session_id`
- `snapshot_id`
- `supi`
- `thread_id`

LLM Agent 并不是在“裸上下文”下运行，而是带着运行时身份执行。

## Runtime, Storage, and Observability

### Runtime Directories

运行期文件主要落在 `runtime/`：

- `runtime/agents/<agent>/cache`
- `runtime/agents/<agent>/work`
- `runtime/agents/<agent>/logs`
- `runtime/interfaces/<source>__<target>/requests`
- `runtime/interfaces/<source>__<target>/responses`
- `runtime/queues/<agent>/`

这使得 Agent 之间的输入输出天然可落盘、可调试、可回溯。

### Database Records

`runtime_store.py` 和 `db_tool.py` 会把以下信息持久化：

- session 控制信息
- session context
- artifact 元数据
- stage result
- agent handoff
- episodic experience
- network snapshot / graph snapshot
- UE context

这说明系统既有“文件级协作面”，也有“数据库级审计面”。

### Knowledge Base

知识库部分由：

- [knowledge_tool.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/tools/knowledge_tool.py)
- [langchain_pg.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/database/langchain_pg.py)
- `knowledge_scripts/`

共同支撑。

它不是通用 RAG，而是面向 PCF/URSP 标准对象的专用知识层，供 IEA 等组件在术语解析、对象识别和规范补全时使用。

## Test Evidence

测试文件表明，当前项目不仅关心“结果能不能跑”，也在验证运行时契约。

### `tests/test_system_coordinator.py`

验证点包括：

- memory context 是否进入 IEA
- snapshot metadata 是否被正确绑定到 OSA/PDA
- IEA/OSA 是否支持瞬时失败后的重试
- 多轮闭环是否会把 feedback 传回下一轮

### `tests/test_agent_runtime.py`

验证点包括：

- Agent workspace 是否隔离
- artifact request/response 是否原子写入
- 不同接口目录是否并行分离
- IEA/OSA/PDA 是否正确缓存收发 artifact

### `tests/test_policy_execution_components.py`

验证点包括：

- `PolicyCompiler` 是否正确抽取 `flow_id`
- `PolicyGuard` 是否拒绝不一致或不完整的 policy
- `AssuranceEvaluator` 是否使用绑定 snapshot，而不是漂移状态

### `tests/test_iea_resolution.py`

验证点包括：

- flow/app 唯一匹配
- 歧义返回候选集
- 无匹配返回 `unmatched`
- 未指定方向时的带宽归一规则
- `OperationIntent` 后处理规范化

这些测试说明系统当前重点不只是“LLM 出不出结果”，而是“结构化协作链条是否可控”。

## Notable Constraints and Risks

### 1. README 与运行时主线不完全对齐

README 主要介绍了知识库脚本，但仓库实际核心是闭环多智能体控制系统。对新读者来说，这会造成理解偏差。

### 2. CR 当前不参与硬阻断

冲突检测已经存在，但它更像旁路分析器而不是强制闸门。若希望把 CR 变成真正的执行前保护，需要在 Coordinator 中把 CR 结果接入提交判定。

### 3. PDA 的执行成功依赖外部 PCF ack 和 snapshot 数据质量

系统内部对策略格式约束很严格，但外部依赖仍然包括：

- PCF 返回格式
- snapshot 是否存在且包含目标 flow
- UE context 是否能成功持久化

### 4. MemoryManager 使用 LLM 做长期记忆摘要

这增强了跨轮对话连续性，但也意味着 long-term memory 不是纯确定性过程。

### 5. 知识库是领域增强层，不是主控层

知识库很重要，但它服务于意图理解和规范补全，不能替代对 UE catalog、snapshot、policy schema 的严格绑定。

## Source Pointers

- 总控入口：[system_coordinator.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/system_coordinator.py)
- Agent 协作契约：[collaboration.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/domain/collaboration.py)
- 核心策略对象：[policy_plan.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/domain/policy_plan.py)
- 策略编译：[policy_compiler.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/domain/policy_compiler.py)
- 策略校验：[policy_guard.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/domain/policy_guard.py)
- 执行控制器：[execution_controller.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/workflows/execution_controller.py)
- 运行时注册表：[runtime_registry.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/workflows/runtime_registry.py)
- IEA：[agents/intent_encoding/agent.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agents/intent_encoding/agent.py)
- OSA：[agents/optimization_strategy/agent.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agents/optimization_strategy/agent.py)
- PDA：[agents/policy_dispatch/agent.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agents/policy_dispatch/agent.py)
- CR：[agents/conflict_resolution/agent.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agents/conflict_resolution/agent.py)
- AD：[agents/assurance_diagnosis/agent.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agents/assurance_diagnosis/agent.py)
- Agent runtime：[artifacts.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agent_runtime/artifacts.py)
- Runtime workspace：[workspace.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agent_runtime/workspace.py)
- Runtime queue：[queue.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/agent_runtime/queue.py)
- 运行时持久化：[runtime_store.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/tools/runtime_store.py)
- 数据库访问：[db_tool.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/tools/db_tool.py)
- 网络图快照：[network_graph.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/tools/network_graph.py)
- 标准知识检索：[knowledge_tool.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/tools/knowledge_tool.py)
- 向量存储：[langchain_pg.py](C:/Users/xiezhengyi/Desktop/research/6G+AI/code/MuiltiAgents/database/langchain_pg.py)

## One-Sentence Summary

这个项目本质上是一个“以 session/snapshot/artifact 为主线、以 IEA-OSA-PDA 为执行核心、以 CR/AD 为旁路分析、以标准知识库为语义增强”的 5G PCF 闭环多智能体策略控制系统。
