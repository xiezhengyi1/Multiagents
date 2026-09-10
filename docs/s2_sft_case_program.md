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

`flow` 不存在、SBA 查询失败、字段不匹配、模型结构输出错误、optimizer 不可用、HTTP 超时和执行回执失败都不是无解样本；它们必须保留为 `failed` 诊断样本，不能进入正向 SFT。

## 公开资料的使用边界

- [3GPP 的 URSP 技术说明](https://www.3gpp.org/images/newsletters/3GPP_Highlights_Issue_10_WEB.pdf)用于约束 URSP 的路由对象和 PCF/AMF 的职责；[23.503 变更记录](https://portal.3gpp.org/DesktopModules/CRs/CrDetails.aspx?CrId=493137)可用于补充时间窗、位置等 Route Selection Descriptor 条件。它们是规范依据，而不是可直接训练的 action-label 数据。
- [SPEC5G](https://arxiv.org/abs/2301.09201)提供公开 5G 规范语言语料，可用于抽取用户请求的多种自然语言表达和术语覆盖，不能作为本项目执行策略的金标。
- [RANalyzer-Dataset](https://github.com/wineslab/RANalyzer-Dataset)包含 OAI 端到端运行、iPerf3 和 gNB 日志，可用于性能 KPI、回归与超时场景的条件分布；不替代 subscriber/URSP 的金标。
- [ORANSlice](https://github.com/wineslab/ORANSlice)展示了 OAI 上的端到端 RAN slicing 实现，可为 RAN slice 配额、负载和观测后验设计 case；其控制接口与本测试床不同，不能直接拷贝策略。

每条保留入库的 SFT 样本必须附带：输入快照版本、观测证据 ID、期望标签、实际 terminal、验证器事件 ID、执行回执或无解证明。这样可审计地筛除偶然成功、错误拒绝和环境不稳定样本。
