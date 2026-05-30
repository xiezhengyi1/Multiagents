# 面向6G核心网策略自治的多智能体闭环控制框架：期刊论文详细大纲

## 0. 论文定位

### 建议标题

**英文题目**  
Multi-Agent Closed-Loop Policy Control with Large Language Models for Intent-Driven 6G Core Networks

**中文题目**  
面向意图驱动6G核心网的基于大语言模型的多智能体闭环策略控制框架

### 论文核心主张

本文不把大语言模型直接作为无约束的核心网控制器，而是将其限定在高层语义理解、对象接地、策略规划、异常归因和协作编排等上层自治任务中；同时由确定性模块负责策略编译、冲突检查、执行下发、状态回写与指标验证。论文重点突出三点：

1. 自然语言控制请求到核心网策略对象之间的可验证映射；
2. 多智能体分工、RAG证据、网络状态快照和优化器之间的协同机制；
3. 执行反馈、保障监控和运行轨迹沉淀形成的闭环自治与数据飞轮。

### 建议投稿方向

首选可面向 **IEEE Transactions on Network and Service Management** 或 **Computer Networks**，因为本文更偏网络管理、策略控制、闭环自治和系统实验；若压缩数学细节并强化综述性表达，也可改写为 **IEEE Wireless Communications** 或 **IEEE Communications Magazine** 风格。

## 1. Abstract

摘要建议控制在200到250词，按四句话展开：

1. 背景：6G核心网需要从静态策略配置走向意图驱动、闭环自治和可验证控制。
2. 问题：LLM可理解自然语言，但在核心网场景中存在对象接地不稳定、结构化策略生成不可靠、跨域约束难保持一致、执行失败后难以定向修正等问题。
3. 方法：提出一个多智能体闭环策略控制框架，由Main、Intent Encoding、Optimization Strategy和Policy Dispatch等角色协作，结合RAG、网络图快照、结构化中间意图、优化器、冲突仲裁、执行诊断和监控重入。
4. 结果：在S1/S2/S3及公开数据集驱动变体上，围绕切片迁移、QoS调整、资源冲突和多对象歧义任务验证，指标包括SGC、PGR、DSR、CRR、TCR以及平均轮次和重试次数。

这里不要把“持续自治监控”写成已经完全生产化部署，应表述为框架中的监控重入能力和可扩展自治闭环原型。

## 2. Index Terms

6G core network, intent-driven networking, large language model, multi-agent system, policy control function, closed-loop control, retrieval-augmented generation, network slicing, QoS control, autonomous network management

## 3. Section I: Introduction

### 3.1 Background and Motivation

本节说明6G核心网为什么需要意图驱动策略控制。可以从以下角度写：

1. 6G核心网将承载远程驾驶、工业机器人、云渲染、远程医疗、AR游戏等差异化业务；
2. 这些业务在时延、带宽、可靠性、切片隔离、移动性连续性方面存在强差异；
3. 传统规则配置和人工运维难以处理多业务、多切片、多UE、多策略域耦合；
4. 自然语言请求例如“把Telemedicine视频流调稳一点，不要影响别的医疗业务”必须被映射为明确的UE、App、Flow、QoS目标、切片目标和策略域边界。

**建议插图 Fig. 1：问题场景图**  
画一个从“自然语言意图”到“核心网策略控制”的场景图：左侧是用户请求，中间是LLM多智能体控制层，右侧是PCF/SM Policy/AM Policy/切片/QoS Flow。图中突出三类困难：语义歧义、对象绑定、跨域约束。

### 3.2 Why LLMs Are Useful but Insufficient

本节既要肯定LLM，也要划清边界：

1. LLM适合语义解析、上下文推理、工具调用决策、异常解释；
2. LLM不适合直接跳过校验生成任意核心网底层流程；
3. 注册、鉴权、PDU Session建立、基础信令流程等受3GPP标准约束，不能由LLM随意重写；
4. 本文关注的灵活空间是策略和编排层：QoS、切片选择、优先级、资源重分配、AM policy边界内的可配置控制。

### 3.3 Research Challenges

建议归纳为五个挑战：

1. **Intent-to-object grounding**：自然语言中的业务名、模糊指代和否定约束需要稳定落到SUPI、App、Flow和策略域。
2. **Domain boundary control**：请求可能只涉及QoS，也可能涉及Mobility；错误扩大或缩小控制域都会造成误操作。
3. **Cross-domain constraint consistency**：QoS目标、切片资源、移动性目标、业务优先级之间存在耦合。
4. **Deterministic execution under uncertain generation**：LLM输出必须经过结构化契约、编译、冲突检查和执行回执验证。
5. **Feedback-directed recovery**：失败后不能简单重试，而应判断是接地失败、规划失败、执行失败还是保障失败，并回流到不同阶段。

### 3.4 Contributions

建议写成四条：

1. 提出一个面向6G核心网策略控制的多智能体闭环框架，将自然语言请求分解为主控路由、意图接地、策略规划、冲突仲裁、策略执行、执行诊断和监控重入。
2. 设计证据化意图接地机制，结合RAG、业务目录、UE上下文、网络图快照和QoS envelope，将语义目标映射为OperationIntent。
3. 引入约束感知策略规划与确定性执行链路，利用优化器预览、PolicyPlanDraft、冲突仲裁和PCF适配层降低LLM自由生成风险。
4. 构建可复现实验资产与轨迹数据管线，在S1/S2/S3和公开数据集画像变体上验证多智能体、RAG、闭环修正对SGC/PGR/DSR/CRR/TCR的影响。

## 4. Section II: System Scope and Problem Formulation

这一节建议新增。原大纲直接进入“LLM如何支持6G”，但本项目更需要先明确控制边界，否则审稿人会质疑LLM是否在改写核心网底层流程。

### 4.1 Controllable Scope in Core Network Policy Management

写清楚两类边界：

1. 不可随意改变：注册、鉴权、PDU Session建立、核心网网元职责和基础信令顺序；
2. 可灵活优化：业务流QoS、切片选择、优先级、AM policy配置、资源冲突处理、监控触发后的局部重规划。

**建议插图 Fig. 2：核心网控制边界图**  
用分层图表达：底层是标准信令流程，上层是策略控制与自治编排。LLM只进入上层策略和编排空间，确定性模块守住底层接口和策略契约。

### 4.2 Task Definition

形式化定义输入、状态和输出：

1. 输入为自然语言请求 \(u_t\)、当前网络快照 \(S_t\)、知识证据 \(K\)、历史反馈 \(M_t\)；
2. 输出为可执行策略集合 \(P_t\)、执行结果 \(F_t\)、更新后的状态 \(S_{t+1}\) 和必要的重试路由；
3. 目标是最大化任务完成率，同时降低错误对象绑定、错误域扩张和无效重试。

可给出简洁公式：

\[
P_t = \mathcal{F}_{plan}(\mathcal{F}_{ground}(u_t, S_t, K, M_t), S_t)
\]

\[
S_{t+1}, F_t = \mathcal{F}_{exec}(\mathcal{F}_{verify}(P_t, S_t))
\]

### 4.3 Task Categories

结合当前 `task_catalog.json`，写四类任务：

1. 切片迁移：例如把Remote_Drive视频流迁移到低时延切片；
2. QoS调整：例如提高Telemedicine下行带宽并降低时延；
3. 资源冲突处理：例如资源紧张时优先保障工业控制流；
4. 多对象歧义消解：例如“robot业务”不能误绑定到普通IoT传感器。

**建议表 Table I：任务类型与代表性请求**  
列出四类任务、自然语言例子、关键挑战、期望输出对象。

## 5. Section III: Proposed Multi-Agent Closed-Loop Framework

这是全文核心章节。

### 5.1 Overall Architecture

描述项目中的主链路：

1. Main Control Agent：负责请求域判断、轮次策略、重试范围、路由决策；
2. Intent Encoding Agent：负责UE/App/Flow/Mobility对象接地，输出OperationIntent；
3. Optimization Strategy Agent：负责策略草案、优化器预览、PolicyPlanDraft；
4. Policy Dispatch / Deterministic Runtime：负责冲突仲裁、策略编译、PCF适配、执行回执；
5. Diagnosis and Monitoring：负责执行归因、保障评估、监控告警和重入触发；
6. Trace and Training Layer：负责记录运行轨迹并投影为训练样本。

**建议插图 Fig. 3：总体框架图**  
用四层结构画：意图入口层、智能体协作层、确定性执行层、状态/知识/训练支撑层。箭头体现闭环：执行反馈回Main，监控告警也能重新生成Requirement并进入Main。

### 5.2 Main Agent: Routing and Round-Level Control

写Main Agent的职责边界：

1. 只做高层路由，不直接做App/Flow最终接地；
2. 输出 `requested_domains`、`next_agent`、`round_strategy`、`retry_scope`、`reuse_contract`；
3. 在重试轮判断进入 `intent_encoding` 还是 `optimization_strategy`；
4. 保证否定约束有效，例如“不要改mobility”时不能加入mobility域。

这一节可以强调Main的创新点不在“更会生成策略”，而在维护阶段边界、重试入口和复用契约。

### 5.3 Intent Encoding Agent: Evidence-Based Grounding

写IEA如何把自然语言映射为OperationIntent：

1. 从用户请求抽取显式SUPI、App名、Flow名、QoS目标、Mobility目标；
2. 调用业务目录、UE上下文、AM policy候选、语义检索和知识库；
3. 输出 `grounding_evidence`、`flows`、`qos_target_envelopes`、`mobility_intent`；
4. 对resolved flow强制要求同时具备 `flow_id` 和 `app_id`；
5. 对多对象歧义保留候选证据，必要时阻塞或触发重新接地。

**建议插图 Fig. 4：意图接地流程图**  
从User Input进入，分三路检索：业务目录、网络快照、RAG知识；汇总到IntentEvidence；再生成OperationIntent。图中标出“证据化接地”和“禁止凭空生成ID”。

### 5.4 Optimization Strategy Agent: Constraint-Aware Policy Planning

写OSA如何生成策略：

1. 输入为OperationIntent和PlanningContext；
2. 利用优化器集成模块处理QoS/Mobility/联合控制；
3. 输出PolicyPlanDraft，包含 `planning_status`、`optimizer_result`、`all_policies`、`missing_evidence`、`planner_conflicts`；
4. 当证据不足时返回 `needs_upstream_reground`，而不是生成不可执行策略；
5. 规划结果要保持和IEA的grounding basis、Main的domain boundary一致。

**建议插图 Fig. 5：策略规划与优化器交互图**  
画OperationIntent、PlanningContext、Optimizer Preview、PolicyPlanDraft之间的数据流，突出“LLM生成候选 + 优化器校验/预览 + 结构化策略草案”。

### 5.5 Conflict Mediation and Deterministic Execution

这一节强调LLM之后的确定性保护：

1. `build_conflict_request_payload` 构建资源视图，包括切片容量、当前负载、flow分配；
2. 仲裁器检查资源冲突、域冲突、策略对象冲突；
3. 策略编译器将PolicyDraft转换为SM Policy或AM Policy请求；
4. PCF dispatch层返回 `dispatch_receipts`；
5. assurance evaluator检查执行后指标是否满足目标。

**建议插图 Fig. 6：确定性执行链路**  
PolicyPlanDraft -> Conflict Request -> Domain Verdict -> Policy Compiler -> PCF Dispatch -> Assurance Verdict -> Diagnosis。

### 5.6 Diagnosis, Retry Routing, and Monitoring Reentry

写闭环如何形成：

1. 执行诊断输入包括execution feedback、dispatch receipts、assurance verdicts、telemetry snapshot；
2. 输出root cause、affected policy、affected flow和recommended actions；
3. Main根据诊断判断full reground、partial reground、target stable或禁止重试；
4. 监控模块可扫描flow指标并生成MonitorAlert；
5. Requirement Agent可把告警转成新的控制请求，触发AutonomousMonitorReentryLoop。

**建议插图 Fig. 7：失败类型与回流路径图**  
左侧列出失败类型：对象接地失败、域边界错误、规划证据不足、冲突仲裁失败、执行失败、保障违约；右侧连接回Main、IEA或OSA。

**建议表 Table II：失败类型、观测症状和修正入口**  
这是论文里很有价值的一张表，能体现“定向修正”区别于简单重试。

## 6. Section IV: Knowledge, State, and Trace Infrastructure

这一节把支撑层写清楚，避免系统看起来只是prompt拼接。

### 6.1 Knowledge Runtime and RAG

写知识构建模块：

1. 从3GPP/PCF相关资料构建知识条目；
2. 支持语义检索、精确检索、ColBERT索引、pgvector迁移；
3. 为IEA和OSA提供术语、策略字段、接口模式和对象映射证据；
4. RAG不是替代状态读取，而是补充领域知识边界。

**建议插图 Fig. 8：知识构建与检索链路**  
资料文档 -> 知识构建脚本 -> 向量库/索引 -> knowledge tool -> IEA/OSA。

### 6.2 Network Graph Snapshot and Scenario Runtime

写网络状态建模：

1. 场景YAML定义UE、gNB、UPF、slice、app、flow、free5GC/ns-3桥接信息；
2. 初始化脚本把场景转为运行态快照；
3. 快照中包含切片容量、负载、业务流分配、UE上下文、拓扑关系；
4. 图快照支持复现实验和跨方法公平比较。

**建议插图 Fig. 9：网络图快照模型**  
节点包括UE、App、Flow、Slice、UPF、gNB；边表示承载、分配、连接、服务关系；节点属性放QoS和资源指标。

### 6.3 Memory and Runtime Trace

写运行时和训练层：

1. agent_runtime记录artifact、queue、trace、workspace和结构化工具循环；
2. 每轮保存GlobalControlIntent、OperationIntent、PolicyPlanDraft、dispatch receipts和diagnosis；
3. training模块将workflow trajectories和agent trajectories投影为训练记录；
4. 导出ChatML数据，用于后续监督微调或偏好优化。

**建议插图 Fig. 10：运行-产数-训练数据飞轮**  
控制运行 -> 原始trace -> 轨迹投影 -> 训练样本 -> 模型更新 -> 再部署。注意图注中写“future extensible data flywheel / prototype support”，不要夸大为已完成全自动训练闭环。

## 7. Section V: Experimental Methodology

### 7.1 Experimental Scenarios

基于当前配置写S1/S2/S3和S1P/S2P/S3P：

1. S1：2 UE、1 gNB、1 UPF、1 slice，适合轻量QoS和语义解析验证；
2. S2：6 UE、多App、多Flow、2到3个slice，适合主对比和消融；
3. S3：10 UE、5 gNB、4 UPF、5 slice，适合复杂场景鲁棒性；
4. S1P/S2P/S3P：保持对象拓扑，替换为公开数据集代理流量画像。

**建议表 Table III：实验场景参数**  
列场景、UE数量、gNB数量、UPF数量、slice数量、flow特征、用途。

### 7.2 Methods and Baselines

写六种方法：

1. B1：SingleAgent，无RAG，无闭环；
2. B2：SingleAgent，有RAG，无闭环；
3. B3：SingleAgent，有RAG，有多轮闭环；
4. Ours：MultiAgent，有RAG，有闭环；
5. Ours w/o RAG：去除RAG；
6. Ours w/o ClosedLoop：只保留首轮，不进行闭环修正。

**建议表 Table IV：方法矩阵**  
列agent数量、RAG、closed loop、multi-round revision、runner。

### 7.3 Evaluation Metrics

采用当前指标脚本中的指标：

1. **SGC**：Semantic Grounding Consistency，最终轮OperationIntent、plan和receipt的域与对象是否一致；
2. **PGR**：Policy Generation Rate，是否生成approved policies；
3. **DSR**：Dispatch Success Rate，dispatch receipt是否success、APPLIED、COMPLIANT；
4. **CRR**：Closed-loop Recovery Rate，首轮失败后最终恢复成功的比例；
5. **TCR**：Task Completion Rate，任务最终成功、完成、下发成功且语义绑定一致；
6. 辅助指标：平均轮次、平均重试次数、执行时间、失败类型分布。

**建议表 Table V：指标定义**  
给出每个指标的计算条件，最好对齐 `compute_thesis_metrics.py`。

## 8. Section VI: Experimental Results and Analysis

### 8.1 Overall Comparison on S2 and S3

写主结果：

1. S2中Ours在SGC/PGR/DSR/TCR上显著高于B1/B2/B3；
2. S3中B3性能大幅下降，而Ours仍保持更高SGC、PGR、DSR和TCR；
3. 说明复杂拓扑下，单智能体闭环容易陷入无方向重试，多智能体分工能更好地定位失败阶段。

可以使用当前结果中的代表值：

| Method | Scenario | SGC | PGR | DSR | CRR | TCR |
|---|---|---:|---:|---:|---:|---:|
| B1 | S2 | 0.3125 | 0.3125 | 0.3125 | 0.0000 | 0.2500 |
| B2 | S2 | 0.3125 | 0.3750 | 0.3750 | 0.0000 | 0.3125 |
| B3 | S2 | 0.5000 | 0.5000 | 0.5000 | 0.1818 | 0.4375 |
| Ours | S2 | 0.8125 | 0.8750 | 0.8750 | 0.3333 | 0.8125 |
| B3 | S3 | 0.2941 | 0.1765 | 0.1765 | 0.0667 | 0.1765 |
| Ours | S3 | 0.6471 | 0.6471 | 0.6471 | 0.5000 | 0.5294 |

**建议插图 Fig. 11：S2/S3主对比柱状图**  
每个场景一组柱状图，指标用SGC/PGR/DSR/TCR，CRR可单独画。

### 8.2 Ablation: Impact of Closed-Loop Correction

写闭环消融：

1. 移除闭环后，CRR为0；
2. TCR下降，说明不少任务依赖第二轮定向修正；
3. 平均轮次降低并不代表更好，因为它可能只是失败后停止。

**建议插图 Fig. 12：闭环消融结果**  
Ours vs Ours w/o ClosedLoop，重点显示CRR和TCR。

### 8.3 Ablation: Impact of RAG

写RAG消融要谨慎，因为不同基座模型表现不同：

1. Qwen配置下，RAG对SGC和TCR提升明显；
2. DeepSeek配置下，Ours w/o RAG在部分S2结果中并不低，说明更强模型或更稳定状态证据可能削弱RAG边际收益；
3. 结论应写成“RAG是与基座模型和状态证据互补的机制”，不要写成所有情况下都单调提升。

**建议插图 Fig. 13：RAG消融双模型对比**  
横轴为Qwen/DeepSeek，纵轴为SGC、TCR，比较Ours和Ours w/o RAG。

### 8.4 Interaction Cost and Recovery Efficiency

用平均轮次和平均重试次数说明效率：

1. S3中B3平均轮次约2.7647，平均重试1.7647；
2. Ours平均轮次约2.2353，平均重试1.2353；
3. Ours在更少交互代价下取得更高TCR，说明定向重试优于盲目多轮。

**建议插图 Fig. 14：成功率与交互代价联合图**  
左轴TCR，右轴平均重试次数；或用散点图，横轴平均重试，纵轴TCR。

### 8.5 Case Study

选择2到3个典型任务写案例：

1. 切片迁移案例：Remote_Drive或Telemedicine迁移到低时延/高可靠切片；
2. 多对象歧义案例：Factory_Robot不要误识别为IoT_Sensor；
3. 域边界案例：用户明确“不改mobility”时，Main和IEA如何保持QoS-only。

每个案例按五步写：

1. 用户请求；
2. Main输出域和路由；
3. IEA接地证据；
4. OSA策略和优化器依据；
5. 执行结果与反馈修正。

**建议插图 Fig. 15：单案例闭环轨迹图**  
用时序图展示round 1失败、diagnosis、round 2修正、最终成功。

## 9. Section VII: Discussion

### 9.1 Why Multi-Agent Collaboration Helps

写多智能体优势：

1. Main限制域和重试入口；
2. IEA专注对象接地；
3. OSA专注策略可行性；
4. Dispatch和Diagnosis提供确定性反馈；
5. 每个阶段的中间产物可检查、可复用、可训练。

### 9.2 Reliability Boundaries

写系统可靠性的边界：

1. 不能保证LLM永不误判；
2. 通过结构化契约和执行前校验降低风险；
3. 对证据不足场景应阻塞或重接地，而不是强行执行；
4. 高风险动作应接入沙箱、回滚或人工审批。

### 9.3 From Reactive Control to Autonomous Assurance

结合项目中的monitoring模块写未来扩展：

1. 当前主实验以用户请求触发为主；
2. 监控重入模块已经具备把FlowTelemetryMonitor告警转为RequirementDraft并重新进入编排器的原型；
3. 后续可把一次性策略生成扩展为持续意图保障；
4. 持续运行trace可进一步形成训练数据飞轮。

## 10. Section VIII: Open Issues and Future Work

建议写五点：

1. **Long-term intent contract**：从单次请求转为长期意图契约，处理生命周期、优先级和冲突覆盖。
2. **Production-grade telemetry and rollback**：接入更真实的遥测、策略回滚和执行审计。
3. **Skill-based orchestration**：从固定流水线进一步升级为可组合的协作skill和协议skill。
4. **Trace-driven model improvement**：利用运行trace进行SFT、偏好学习和失败样本挖掘。
5. **Cross-domain scaling**：扩展到RAN、MEC、传输网和跨域资源编排。

## 11. Section IX: Conclusion

结论建议围绕三句话：

1. 本文提出了一个面向6G核心网策略控制的LLM多智能体闭环框架；
2. 该框架通过证据化接地、约束化规划、确定性执行和反馈驱动修正，将自然语言意图转化为可验证策略动作；
3. 实验表明，多智能体闭环在中高复杂度场景下优于单智能体基线，并为未来持续意图保障和在线数据飞轮奠定了工程基础。

## 12. 推荐图表总表

| 编号 | 图表内容 | 放置章节 | 作用 |
|---|---|---|---|
| Fig. 1 | 自然语言到核心网策略控制的问题场景 | Introduction | 说明研究问题 |
| Fig. 2 | 核心网标准流程与策略可配置边界 | System Scope | 回应LLM控制边界 |
| Fig. 3 | 多智能体闭环总体框架 | Framework | 论文主图 |
| Fig. 4 | 意图接地与证据融合流程 | Framework | 展示IEA创新 |
| Fig. 5 | OSA与优化器交互流程 | Framework | 展示规划机制 |
| Fig. 6 | 冲突仲裁与确定性执行链路 | Framework | 展示安全执行 |
| Fig. 7 | 失败类型与定向回流路径 | Framework | 展示闭环修正 |
| Fig. 8 | RAG知识构建与检索链路 | Infrastructure | 展示知识底座 |
| Fig. 9 | 网络图快照模型 | Infrastructure | 展示状态表示 |
| Fig. 10 | 运行trace到训练数据飞轮 | Infrastructure/Discussion | 展示长期价值 |
| Fig. 11 | S2/S3主对比结果 | Results | 展示总体效果 |
| Fig. 12 | 闭环消融结果 | Results | 展示闭环收益 |
| Fig. 13 | RAG消融双模型对比 | Results | 展示条件性收益 |
| Fig. 14 | TCR与平均重试次数对比 | Results | 展示效率 |
| Fig. 15 | 典型案例闭环轨迹 | Results | 展示可解释过程 |
| Table I | 任务类型与代表性请求 | Problem Formulation | 说明任务集 |
| Table II | 失败类型与修正入口 | Framework | 说明定向修正 |
| Table III | 场景参数 | Experiments | 说明实验环境 |
| Table IV | 方法矩阵 | Experiments | 说明baseline |
| Table V | 指标定义 | Experiments | 对齐代码指标 |
| Table VI | 主实验结果 | Results | 汇总核心性能 |
| Table VII | 消融实验结果 | Results | 汇总模块贡献 |

## 13. 写作注意事项

1. 不要把系统写成“LLM直接控制核心网底层流程”，应始终强调标准流程边界和确定性执行保护。
2. 实验指标优先使用 `SGC/PGR/DSR/CRR/TCR`，这与当前 `compute_thesis_metrics.py` 对齐。
3. S1只适合作为轻量补充，不适合作为主实验结论来源；主对比应以S2/S3为主。
4. RAG结论要写成条件性收益，避免和现有DeepSeek消融结果冲突。
5. “在线自生长数据集”和“持续自治保障”是重要创新方向，但当前论文中应表述为已有原型支撑和未来扩展，而不是完全成熟生产系统。
