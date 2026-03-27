可以，已整理成可直接落地的方案文档内容。  
当前会话处于只读模式，我不能直接在仓库里新建文件；你可以把下面内容保存为例如 docs/ibns_cn_ran_context_reduction_plan.md（若该路径尚不存在，可放到已有目录）。

# IBNS-CN/RAN 上下文减负与全量数据注入方案

## 1. 背景与目标
现状：
- 智能体工具在调用联合仿真时要求 app_details 携带 flows、nodes、service_types、sla_profiles，导致上下文负担重。
- 网络状态查询原本偏摘要或单一形式，难以同时满足“轻量感知”和“重仿真输入”。

目标：
- 将网络状态查询拆分为摘要与全量两类。
- 智能体仅使用摘要进行决策引导。
- 联合仿真工具内部自动读取全量状态并拼装 app_details，减少大模型传参负担。
- 明确 sla_profiles 的来源与回退策略。

## 2. 关键设计

### 2.1 双查询函数
在 network_status.py 提供：
- get_network_status_summary  
  用途：给智能体看概况。  
  内容：切片状态（用量、type、关键指标）+ apps 简表。
- get_network_status_full  
  用途：给仿真工具构造输入。  
  内容：可直接映射为 run_joint_simulation 所需字段：
  - flows_data
  - nodes_data
  - service_types_data
  - sla_profiles_data

兼容性建议：
- 保留 get_network_status 作为别名指向全量函数，避免旧调用立即失效。

### 2.2 智能体调用策略
在 OptimizationStrategyAgent.py：
- fetch_network_status 仅调用 get_network_status_summary。
- self.tools 保持最小工具集合，避免无关数据进入上下文。
- run_joint_cn_ran_simulation 的 app_details 改为“增量意图”为主，不再要求传 nodes。

### 2.3 联合仿真工具入参拼装
在 OptimizationStrategyAgent.py 的 run_joint_cn_ran_simulation 中：
1. 内部调用全量状态函数或场景读取函数，得到基线环境。
2. 从 app_details 读取增量 flows/service_types/sla_profiles。
3. 合并策略：
- flows = baseline_flows + incremental_flows
- nodes = baseline_nodes（通常不允许模型覆盖）
- service_types = baseline_service_types 被 app_details 同 id 覆盖
- sla_profiles = 按优先级决策（见下节）
4. 调用 run_joint_simulation。

## 3. app_details 新约定

### 3.1 推荐最小输入
app_details 只需包含新增业务意图：
- flows（必需）
- service_types（可选）
- sla_profiles（P2 可选但推荐）

### 3.2 不再要求
- nodes：由工具自动从全量状态读取。

## 4. sla_profiles 来源与优先级

数据库现状：
- 未见独立 SLAProfile 业务表；
- UE 上下文中有 sla_profile JSON 字段，见 models.py。

建议优先级：
1. app_details.sla_profiles（显式传入优先）
2. UE 上下文聚合生成（ue_context.sla_profile）
3. 默认模板回退（按 service_type_id 生成保守阈值）

P2 模式策略：
- 若最终无 sla_profiles：记录 warning，并可二选一：
  - 严格模式：直接返回错误，提示必须提供 SLA
  - 宽松模式：自动套用默认模板并继续

## 5. 文档与注释同步点

需要同步修改的说明文本：
- OptimizationStrategyAgent.py 中 run_joint_cn_ran_simulation 的 Args 注释，删除 nodes 必填描述，改为“自动读取全量环境”。
- network_status.py 的函数注释，区分摘要/全量用途。
- 若有对外接口文档，也同步 app_details 最小结构。

## 6. 验收标准

功能验收：
- 智能体首次只拉取摘要状态，不出现大体积 nodes/slices 全量文本。
- 仿真工具在仅提供新增 flows 时可成功运行。
- P2 下 sla_profiles 缺失时行为符合设定策略（报错或默认回退）。

性能验收：
- 智能体单轮上下文 token 显著下降。
- 联合仿真输入数据完整性不下降（nodes、存量 flows、service_types 可用）。

## 7. 实施顺序（建议）
1. 先改 network_status.py 拆分函数并保留兼容别名。  
2. 再改 OptimizationStrategyAgent.py 的摘要调用与仿真工具拼装逻辑。  
3. 最后补充 sla_profiles 聚合与回退策略，并更新注释和测试用例。

如果你要，我下一步可以给你一版“可直接粘贴”的最终 Markdown 成稿（带目录、版本号、变更记录模板）。