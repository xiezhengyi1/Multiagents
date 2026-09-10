# S2 无解终态与 SFT case 计划

## 已落地的终态规则

每个 terminal 都必须写入以下字段：

| 字段 | 含义 | 可进入 SFT |
| --- | --- | --- |
| `outcome=success`, `resolution=executed` | 策略已被执行并通过结果评估 | 是 |
| `outcome=success`, `resolution=infeasible` | 三智能体链路用完整证据证明请求在当前授权/能力边界内无解 | 是 |
| `outcome=failure`, `resolution=failed` | 结构校验、证据、编译、执行或预算问题 | 否 |

`reason_code` 是稳定的机器标签，`reason` 保留面向人的完整解释。SFT 投影只选择 `outcome=success` 的回合，并把 `resolution` 与 `reason_code` 写到每条样本中。

“无解”不能由模型一句话声明。当前首先认证 `blocked_by_subscription_entitlement`，且必须同时满足：SBA 有订阅查询结果；无任何授权目标 S-NSSAI；不要求订阅变更；PPA 的 partial plan 没有缺失证据或上游查询；其阻塞目标、冲突码和 partial policy 都精确指向同一订阅授权阻塞。任何一项不满足，仍为可修复或失败，不能标为无解。

对执行闭环，完全相同的执行失败指纹（失败原因和受影响 operation 集）只允许修复一次；第二次出现则以 `repeated_execution_failure` 失败终止。这避免 PPA 重复下发等价计划，也不会把运行时超时伪装成无解。

## S2 回归顺序

1. 用干净测试床全量复测 30 个 S2 case。
2. 以每个 case 的 `results.jsonl.terminal.reason`、`resolution` 和 `reason_code` 归因；`status.jsonl` 只用于运行状态，不能作为语义修复依据。
3. 正确率定义为“实际 `resolution` 与 case 的期望终态一致，且 executed 已通过执行评估 / infeasible 已通过证据认证”。目标至少 75%。
4. 未达标时先只复测上轮失败 case；当失败数小于或等于 7 时，下一轮改为全量 30 case 复测。
5. 每轮只修改提示词或逻辑代码；不改变 Main、SBA、PPA 的工具权限和职责。

## 1000-case SFT 语料方案

每个新 case 采用下列必填标注，生成时写入 case JSONL 的 `metadata`，运行后将实际 terminal 与预期对齐：

```json
{
  "feasibility": "feasible | infeasible",
  "expected_resolution": "executed | infeasible",
  "expected_reason_code": "stable_machine_code",
  "evidence_contract": ["required authoritative observations"],
  "acceptance_checks": ["observable postconditions"],
  "scenario_family": "qos | ursp | mobility | cross_domain | safety"
}
```

建议首版总量为 1,000：600 个可执行 case、400 个不可执行 case。每个模板使用不同 SUPI、流、应用、DNN、S-NSSAI、负载、目标与表达方式生成受控变体；同一语义模板不跨训练、验证和测试集合泄漏。

| 类别 | 数量 | 例子 | 正确终态 |
| --- | ---: | --- | --- |
| QoS 同 PDU 调整 | 180 | 带宽、GBR、时延、抖动、优先级 | `executed` |
| 已有 PDU 间 URSP 选路 | 160 | flow/app 到已订阅 DNN/S-NSSAI | `executed` |
| AM/URSP 联动 | 100 | 明确请求 RFSP/AM 变化且授权可用 | `executed` |
| 多 SUPI 分阶段控制 | 80 | 主 SUPI 与当前 stage 精确对齐 | `executed` |
| 执行安全与冲突修复 | 80 | QoS/AMBR/URSP 不冲突的可执行方案 | `executed` |
| 订阅无授权目标 | 140 | 无不同授权 slice，且未请求订阅变更 | `infeasible` / `blocked_by_subscription_entitlement` |
| 已授权但当前 PDU 池缺目标 | 100 | 动态 PDU 必需、当前边界不创建 | `infeasible` / `dynamic_pdu_required` |
| 测试床能力明确缺失 | 80 | capability inventory 明确不支持的操作 | `infeasible` / capability code |
| 安全约束不可满足 | 80 | 目标违反明确订阅/服务区/冲突约束 | `infeasible` / constraint code |


### 场景与模板目录

1,000 个 case 不是把 30 个 S2 请求改写 33 次。首版固定为 100 个语义模板，每个模板在一个明确的初始场景下展开 10 个受控变体。模板、初始状态和语言表达三者必须同时固定，标签不得由模型生成。

| 模板族 | 模板数 x 变体 | 初始场景 | 操作与覆盖 | 期望终态 |
| --- | ---: | --- | --- | --- |
| F1 QoS 同 PDU | 18 x 10 | 正常、拥塞、竞争负载 | priority、GBR/MBR、时延、抖动、上下行带宽、保护与降级 | `executed` |
| F2 既有 PDU 的 URSP | 16 x 10 | 双 PDU、双 DNN、双 S-NSSAI | flow/app Traffic Descriptor、DNN/S-NSSAI 选路、优先级冲突消解 | `executed` |
| F3 AM/URSP | 10 x 10 | AM association 可用、已有 PDU 池 | RFSP-only、显式 AM+URSP、AM 不触发 PDU 迁移 | `executed` |
| F4 多 SUPI 分阶段 | 8 x 10 | 两至三个 UE、资源竞争 | protect/deprioritize/defer 顺序、逐 SUPI 绑定、负约束 | `executed` |
| F5 安全执行 | 8 x 10 | 负载峰值、既有 UE policy | 预算边界、无冲突 QoS/AMBR、无多余 URSP | `executed` |
| I1 订阅无授权目标 | 14 x 10 | 单授权或无不同授权目标 | 请求未订阅的 slice/DNN，且不请求变更订阅 | `infeasible` / `blocked_by_subscription_entitlement` |
| I2 目标 PDU 不存在 | 10 x 10 | 已授权但 PDU 池不含目标组合 | 目标路由需要新 PDU，而当前边界禁止动态创建 | `infeasible` / `dynamic_pdu_required` |
| I3 能力缺失 | 8 x 10 | capability inventory 明确不支持 | 真实无线重选、未支持 AM 字段或未支持 URSP profile | `infeasible` / `capability_unavailable` |
| I4 安全约束冲突 | 8 x 10 | 服务区/订阅/优先级冲突 | 不允许的服务区、显式禁止迁移、不可同时满足的保护目标 | `infeasible` / `constraint_unsatisfied` |

场景 profile 必须由测试床初始化脚本显式建立，而非让 case 假设状态存在：`qos_normal`、`qos_congested`、`qos_contention`、`dual_pdu_ursp`、`am_rfsp_only`、`am_ursp_authorized`、`multi_supi_staged`、`subscription_denied`、`pdu_pool_gap`、`capability_gap` 与 `safety_conflict`。每个 profile 都要落盘其订阅、PDU 池、capability inventory、负载快照与允许的流目录。
`flow` 不存在、SBA 查询失败、字段不匹配、模型结构输出错误、optimizer 不可用、HTTP 超时和执行回执失败都不是无解样本；它们必须保留为 `failed` 诊断样本，不能进入正向 SFT。

### Case 金标与 SFT 准入

每条 case 除现有 metadata 外，必须有以下不可缺失的金标。`expected_agent_path` 约束职责产物，不授权 agent 使用新工具或承担新职责。

```json
{
  "catalog_id": "S2K-F2-003-v07",
  "scenario_profile": "dual_pdu_ursp",
  "operation_class": "ursp_existing_pdu",
  "feasibility": "feasible",
  "expected_terminal": {"resolution": "executed", "reason_code": "objective_achieved"},
  "expected_agent_path": {"main_domains": ["qos"], "sba_evidence": ["get_sm_ue_flow_catalog"], "ppa_plan": "existing_pdu_only"},
  "evidence_contract": ["snapshot_id", "flow_catalog", "pdu_session_pool"],
  "acceptance_checks": ["receipt_success", "target_flow_route_or_qos_changed"],
  "sft_disposition": "positive"
}
```

不可执行 case 的 `expected_agent_path` 必须包含 Main 的正确域路由、SBA 的权威查询与 PPA 的 `partial_plan`；`missing_evidence`、`upstream_requests` 和 `open_questions` 必须为空。只有实际终态严格等于金标、且证据 contract 完整时，`sft_disposition=positive`。所有其他结果记为 `diagnostic`，即使模型给出的自然语言看似合理。


| 实际结果 | 是否标为“正确” | SFT disposition |
| --- | --- | --- |
| `executed` 且全部 acceptance checks 通过 | 是 | `positive` |
| `infeasible` 且终态、reason code、证据 contract 均匹配 | 是 | `positive` |
| `infeasible` 但缺少权威证据或 reason code 不匹配 | 否 | `diagnostic` |
| 结构、工具、编译、执行、超时、预算或 vLLM transport 错误 | 否 | `diagnostic` |

### 划分、生成与验收

- 训练/验证/测试固定为 700/150/150。完整语义模板只能落在一个 partition：70/15/15 个模板，每模板恰好 10 个变体；不得只改 SUPI、应用名或同义词后跨分区。
- 变体只改变已声明的维度：SUPI/流/应用、负载档、授权集合、PDU 池、目标 KPI、优先级、中文表达和否定约束。每个变体必须同时有可机器检查的前置状态与验收状态。
- 先生成 100-case pilot（每个模板族 10 个），逐条经过环境 preflight、三智能体执行和人工抽审；pilot 的标签与场景合格后才 materialize 余下 900 个。未通过 preflight 的记录不得以虚构无解标签补位。
- 每个可执行 case 需要干净场景、执行前 snapshot、action receipt、执行后 snapshot 和 acceptance checks；每个无解 case 需要 subscription/capability/PDU-pool 的权威证据与三个 agent 的终止链路。

### 后续任务记录

1. 修复并全量复测 S2 30 case；正确率以金标终态匹配计算，达到或超过 75% 才能冻结 S2 基线。
2. 若未达到 75%，依据每条 `results.jsonl.terminal` 的 reason、resolution 和 reason code 迭代提示词或逻辑代码；失败数大于 7 时只复测失败集，失败数不大于 7 时下一轮全量复测。
3. 实现上述 11 个 scenario profile 的初始化与 preflight，并将 100 个模板的金标写为可版本化 catalog；先运行 100-case pilot。
4. pilot 通过后生成其余 900 个 case，执行、审计并仅投影 `positive` 样本到 SFT；`diagnostic` 样本独立保留供错误分析。
## 公开资料的使用边界

- [3GPP 的 URSP 技术说明](https://www.3gpp.org/images/newsletters/3GPP_Highlights_Issue_10_WEB.pdf)用于约束 URSP 的路由对象和 PCF/AMF 的职责；[23.503 变更记录](https://portal.3gpp.org/DesktopModules/CRs/CrDetails.aspx?CrId=493137)可用于补充时间窗、位置等 Route Selection Descriptor 条件。它们是规范依据，而不是可直接训练的 action-label 数据。
- [SPEC5G](https://arxiv.org/abs/2301.09201)提供公开 5G 规范语言语料，可用于抽取用户请求的多种自然语言表达和术语覆盖，不能作为本项目执行策略的金标。
- [RANalyzer-Dataset](https://github.com/wineslab/RANalyzer-Dataset)包含 OAI 端到端运行、iPerf3 和 gNB 日志，可用于性能 KPI、回归与超时场景的条件分布；不替代 subscriber/URSP 的金标。
- [ORANSlice](https://github.com/wineslab/ORANSlice)展示了 OAI 上的端到端 RAN slicing 实现，可为 RAN slice 配额、负载和观测后验设计 case；其控制接口与本测试床不同，不能直接拷贝策略。

每条保留入库的 SFT 样本必须附带：输入快照版本、观测证据 ID、期望标签、实际 terminal、验证器事件 ID、执行回执或无解证明。这样可审计地筛除偶然成功、错误拒绝和环境不稳定样本。
