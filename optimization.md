# 3. 优化问题建模

首先，我们对网络拓扑图 $G=(\mathcal{I},\mathcal{V},\mathcal{L})$ 及其内部的逻辑集合与遥测参数进行定义。

- $\mathcal{A}$：应用集合。
- $\mathcal{F}$：所有业务流集合，且 $\mathcal{F}_i$ 表示应用 $a_i$ 的流集合。
- $\mathcal{S}$：候选切片集合。
- $\mathcal{K}$：物理节点集合。
- $x_{f,j}\in\{0,1\}$：流 $f$ 是否映射到切片 $s_j$。
- $z_f\in\{0,1\}$：流 $f$ 是否被接纳。
- $B_{f,j}^{ul}, B_{f,j}^{dl}\ge 0$：流 $f$ 在切片 $s_j$ 上获得的上下行实际带宽。
- $d_f^{ul}, d_f^{dl}\ge 0$：流 $f$ 的上下行请求缺口。
- $\gamma_f^{ul}, \gamma_f^{dl}\ge 0$：流 $f$ 的上下行 GBR 缺口。
- $\mathrm{dev}_j\ge 0$：切片 $s_j$ 相对目标负载率的偏差。
- $\mathrm{chg}_{f,j}\ge 0$：流 $f$ 在切片 $s_j$ 上相对历史映射的变化量。

其中，流级需求与切片遥测参数定义如下：

- $B_f^{ul}, B_f^{dl}$：流 $f$ 的上下行请求带宽。
- $G_f^{ul}, G_f^{dl}$：流 $f$ 的上下行 GBR。
- $D_f$：流 $f$ 的最大可接受时延。
- $L_f$：流 $f$ 的最大可接受丢包率。
- $J_f$：流 $f$ 的最大可接受抖动。
- $P_f$：流 $f$ 的优先级，数值越小表示优先级越高。
- $V_f=\frac{1}{\max(1,P_f)}$：流 $f$ 的价值权重。
- $\mathrm{SST}(s_j)$：切片 $s_j$ 的标准化业务类型。
- $\mathrm{Type}(f)$：流 $f$ 的业务类型。
- $D_j^{link}$：切片 $s_j$ 的链路时延遥测值。
- $D_j^{proc}$：切片 $s_j$ 的处理时延遥测值。
- $\Lambda_j$：切片 $s_j$ 的丢包率遥测值。
- $\Gamma_j$：切片 $s_j$ 的抖动遥测值。
- $C_j^{ul}, C_j^{dl}$：切片 $s_j$ 的上下行总容量。
- $R_j$：切片 $s_j$ 的保留带宽。
- $U_j^{ul}, U_j^{dl}$：切片 $s_j$ 的当前上下行基础负载。

为便于书写，定义切片的有效剩余容量为：

$$
\bar{C}_j^{ul}=\max(0,C_j^{ul}-R_j-U_j^{ul}),
$$

$$
\bar{C}_j^{dl}=\max(0,C_j^{dl}-R_j-U_j^{dl}).
$$

此外，定义类型兼容指示函数：

$$
\kappa_{f,j}=
\begin{cases}
1, & \mathrm{SST}(s_j)=\mathrm{Type}(f),\\
0, & \text{otherwise}.
\end{cases}
$$

## 一、约束条件

### 1. 切片类型强制匹配约束

在 3GPP 框架内，PDU 会话的建立必须基于合法的网络切片选择辅助信息（S-NSSAI）。应用流的内在业务属性必须与切片的标准化定义相契合。如果流 $f$ 被映射到切片 $s_j$，则其类型必须匹配：

$$
x_{f,j}=1 \Rightarrow \mathrm{SST}(s_j)=\mathrm{Type}(f),\quad \forall f\in\mathcal{F},\forall j\in\mathcal{S}
$$

在线性化后，上式可写为：

$$
x_{f,j}\le \kappa_{f,j},\quad \forall f\in\mathcal{F},\forall j\in\mathcal{S}
$$

该约束保证了 eMBB、URLLC、mMTC 等业务不会被错误地送入语义不匹配的切片。

### 2. 端到端确定性时延保障约束

6G 网络对 IC、HRLLC 等业务提出了极致的时延要求。只有当流 $f$ 被映射到切片 $s_j$ 时，切片的遥测时延才需要满足其要求：

$$
x_{f,j}\cdot (D_j^{link}+D_j^{proc}) \le D_f,\quad \forall f\in\mathcal{F},\forall j\in\mathcal{S}
$$

这里需要指出一个与实现一致的建模修正：当前代码并未将该约束作为硬约束写入可行域，而是将其改造成核心 QoS 软惩罚项。原因很直接，如果将时延、丢包、抖动全部硬约束化，在拥塞态下问题极易无解。因此，当前系统采用“核心时延违约高惩罚，但不立即判死”的策略，以保证在极端压力下仍能输出次优但可执行的解。

对应的时延违约指示量定义为：

$$
q_{f,j}^{lat}=
\begin{cases}
1, & D_j^{link}+D_j^{proc}>D_f,\\
0, & \text{otherwise}.
\end{cases}
$$

### 3. 容量分配与动态挤占/降级约束

网络切片在面临突发海量需求时，资源必然匮乏。系统必须具备意图冲突缓解能力。与传统显式“回收低优先级旧流带宽再重新分配”的写法不同，当前实现采用一种更稳定、也更易于保持可行性的线性建模方式：通过“接纳变量 + 带宽缺口变量 + GBR 缺口变量”来隐式实现拒绝、降级和挤占。

首先，每个流不再被强制完整接纳，而是满足：

$$
\sum_{j\in\mathcal{S}} x_{f,j}=z_f,\quad \forall f\in\mathcal{F}
$$

其中：

- $z_f=1$ 表示流 $f$ 被接纳；
- $z_f=0$ 表示流 $f$ 被拒绝。

其次，分配的实际带宽必须满足非负性、请求上限约束，并且仅在发生映射时存在：

$$
0\le B_{f,j}^{ul}\le x_{f,j}\cdot B_f^{ul},\quad \forall f\in\mathcal{F},\forall j\in\mathcal{S}
$$

$$
0\le B_{f,j}^{dl}\le x_{f,j}\cdot B_f^{dl},\quad \forall f\in\mathcal{F},\forall j\in\mathcal{S}
$$

请求带宽守恒约束为：

$$
\sum_{j\in\mathcal{S}} B_{f,j}^{ul}+d_f^{ul}=B_f^{ul},\quad \forall f\in\mathcal{F}
$$

$$
\sum_{j\in\mathcal{S}} B_{f,j}^{dl}+d_f^{dl}=B_f^{dl},\quad \forall f\in\mathcal{F}
$$

其中 $d_f^{ul},d_f^{dl}$ 表示流 $f$ 的未满足带宽缺口。如果资源不足，系统不会直接无解，而是允许 $d_f^{ul},d_f^{dl}>0$，并在目标函数中对高优先级业务赋予更高惩罚。

同时，GBR 也被建模为软保障：

$$
\sum_{j\in\mathcal{S}} B_{f,j}^{ul}+\gamma_f^{ul}\ge G_f^{ul},\quad \forall f\in\mathcal{F}
$$

$$
\sum_{j\in\mathcal{S}} B_{f,j}^{dl}+\gamma_f^{dl}\ge G_f^{dl},\quad \forall f\in\mathcal{F}
$$

其中 $\gamma_f^{ul},\gamma_f^{dl}$ 表示低于 GBR 的缺口。它们并非完全禁止，而是被高权重惩罚。

对于任何切片 $s_j$，新分配总带宽不得超过其有效剩余物理容量：

$$
\sum_{f\in\mathcal{F}} B_{f,j}^{ul}\le \bar{C}_j^{ul},\quad \forall j\in\mathcal{S}
$$

$$
\sum_{f\in\mathcal{F}} B_{f,j}^{dl}\le \bar{C}_j^{dl},\quad \forall j\in\mathcal{S}
$$

因此，当前系统中的“挤占”并不是通过单独列出“所有低于当前最高优先级流的可牺牲旧流集合”来显式求和，而是通过以下机制隐式完成：

- 低优先级流的体验损失惩罚较小；
- 高优先级流的接纳失败、带宽缺口和 GBR 缺口惩罚更大；
- 在容量约束下，求解器自然会优先牺牲低价值流。

这套约束组合构成了动态意图冲突消解的核心数学基础。

### 4. 物理节点异构计算资源约束

我们将节点资源约束拆解为 VNF 与 MEC 两类独立约束。

#### (4a) VNF 基础信令与转发计算能力约束

任何物理节点 $k\in\mathcal{K}$ 承担的 VNF 计算负载都随新流量线性增长。定义节点 $k$ 托管的切片集合为 $\mathcal{S}(k)$，则节点承载的新业务总流量为：

$$
\Theta_k=\sum_{j\in\mathcal{S}(k)}\sum_{f\in\mathcal{F}}\left(B_{f,j}^{ul}+B_{f,j}^{dl}\right)
$$

对于核心网节点 $k\in\mathcal{K}_{cn}$：

$$
\hat{u}_k^{cpu}\cdot \kappa_k+\alpha_{cn}\Theta_k\le \kappa_k,\quad \forall k\in\mathcal{K}_{cn}
$$

对于接入网节点 $k\in\mathcal{K}_{an}$：

$$
\hat{u}_k^{cpu}\cdot \kappa_k+\alpha_{an}\Theta_k\le \kappa_k,\quad \forall k\in\mathcal{K}_{an}
$$

其中：

- $\kappa_k$ 为节点 CPU 能力上限；
- $\hat{u}_k^{cpu}$ 为当前 CPU 利用率；
- $\alpha_{cn},\alpha_{an}$ 分别表示 CN、AN 的单位流量转发处理开销。

此外，对于接入网节点还存在 PRB 约束：

$$
\hat{u}_k^{prb}\cdot \mathrm{PRB}_k+\pi\Theta_k\le \mathrm{PRB}_k,\quad \forall k\in\mathcal{K}_{an}
$$

其中 $\pi$ 为单位流量对应的 PRB 消耗系数。

#### (4b) MEC 特定业务计算能力约束

对于承载切片 $s_j$ 的节点 $k$，其 MEC 消耗建模为：

$$
\sum_{j\in\mathcal{S}(k)}\mu_{\mathrm{SST}(s_j)}
\sum_{f\in\mathcal{F}}\left(B_{f,j}^{ul}+B_{f,j}^{dl}\right)
\le
(1-\hat{u}_k^{mec})\zeta_k,\quad \forall k\in\mathcal{K}
$$

其中：

- $\mu_{\mathrm{SST}(s_j)}$ 为按切片 SST 区分的 MEC 单位流量开销；
- $\zeta_k$ 为节点 MEC 总能力；
- $\hat{u}_k^{mec}$ 为当前 MEC 利用率。

因此，当 HRLLC 或其他边缘敏感业务被映射到某切片时，系统不仅要满足带宽和时延要求，还必须保证该切片所在节点具有足够的边缘计算承载能力。

### 5. 负载均衡约束

为了避免单一切片过载，我们引入目标负载率 $\rho$。对于切片 $s_j$，其上下行负载率定义为：

$$
\lambda_j^{ul}=\frac{\sum_{f\in\mathcal{F}}B_{f,j}^{ul}+R_j+U_j^{ul}}{C_j^{ul}}
$$

$$
\lambda_j^{dl}=\frac{\sum_{f\in\mathcal{F}}B_{f,j}^{dl}+R_j+U_j^{dl}}{C_j^{dl}}
$$

再通过辅助变量 $\mathrm{dev}_j$ 线性化负载率与目标值之间的绝对偏差：

$$
\mathrm{dev}_j\ge \lambda_j^{ul}-\rho,\quad \forall j\in\mathcal{S}
$$

$$
\mathrm{dev}_j\ge \rho-\lambda_j^{ul},\quad \forall j\in\mathcal{S}
$$

$$
\mathrm{dev}_j\ge \lambda_j^{dl}-\rho,\quad \forall j\in\mathcal{S}
$$

$$
\mathrm{dev}_j\ge \rho-\lambda_j^{dl},\quad \forall j\in\mathcal{S}
$$

这里也必须指出一个与你给出的模板不同的专业修正：模板中使用的是二次型负载均衡项 $\left(\mathrm{LoadRatio}-\rho\right)^2$，而当前代码实现采用的是线性绝对偏差形式。这是因为当前求解器使用的是 CBC 线性整数规划框架，直接引入二次项会把问题从 MILP 推向 MIQP，与现有实现不一致。

### 6. 信令开销约束

设历史映射矩阵为 $x_{f,j}^{old}$，则采用如下线性化形式描述切片重映射代价：

$$
\mathrm{chg}_{f,j}\ge x_{f,j}-x_{f,j}^{old},\quad \forall f\in\mathcal{F},\forall j\in\mathcal{S}
$$

$$
\mathrm{chg}_{f,j}\ge x_{f,j}^{old}-x_{f,j},\quad \forall f\in\mathcal{F},\forall j\in\mathcal{S}
$$

从而有：

$$
\mathrm{chg}_{f,j}=|x_{f,j}-x_{f,j}^{old}|
$$

在最优解处成立。

这一约束使系统不会频繁改变策略矩阵，保障控制面的稳定性。

### 7. 增量与混合模式约束

对于增量优化模式，已有流的切片映射被固定：

$$
x_{f,j}=
\begin{cases}
1, & j=j_f^{old},\\
0, & j\ne j_f^{old},
\end{cases}
\quad \forall f\in\mathcal{F}^{old}
$$

在严格增量模式下，若历史分配带宽存在，则还进一步固定：

$$
B_{f,j_f^{old}}^{ul}=\hat{B}_f^{ul},\quad
B_{f,j_f^{old}}^{dl}=\hat{B}_f^{dl}
$$

但这类强约束可能导致问题无解。因此，当前实现加入了自动可行性回退机制：

- 首先求解严格增量问题；
- 若无解，则保留旧切片映射不变，但释放历史带宽固定约束，转而求解 relaxed incremental。

对于 hybrid 模式，则仅固定旧映射，不固定旧带宽。这使系统能够在同一切片内部主动压缩低优先级流，为高优先级新业务腾挪空间。

## 二、多目标优化函数

目标函数融合了负载均衡、信令开销和体验损失三重核心维度，并额外将丢包/抖动作为辅助 QoS 项独立建模：

$$
\min Z=
\omega_1\underbrace{\frac{1}{|\mathcal{S}|}\sum_{j\in\mathcal{S}}\mathrm{dev}_j}_{\text{负载均衡项}}
+
\omega_2\underbrace{\frac{1}{|\mathcal{F}|}\sum_{f\in\mathcal{F}}\sum_{j\in\mathcal{S}}\mathrm{chg}_{f,j}}_{\text{信令开销项}}
+
\omega_3\underbrace{\left(\Phi_{exp}+\Phi_{qos}^{core}\right)}_{\text{业务体验损失项}}
+
\omega_4\underbrace{\Phi_{qos}^{aux}}_{\text{辅助QoS违约项}}
+
\varepsilon\underbrace{\Phi_{tie}}_{\text{平局打破项}}
$$

### 1. 负载均衡项

权值 $\omega_1$ 控制负载均衡项的重要性。系统并不直接追求“某一切片满载、其他切片空闲”的极端高利用，而是通过惩罚切片负载相对目标负载率 $\rho$ 的偏差，使业务在多个切片间尽量均匀铺开。

这会带来两个直接收益：

- 保持切片运行在更安全的利用区间；
- 为后续突发流量和高价值意图预留缓冲空间。

### 2. 信令开销项

权值 $\omega_2$ 控制网络策略变更频率。通过惩罚新决策矩阵 $x_{f,j}$ 与旧决策矩阵 $x_{f,j}^{old}$ 之间的差异，系统能够抑制频繁重路由与策略抖动，从而保障控制面的稳定性。

### 3. 业务体验损失项

权值 $\omega_3$ 聚焦于最终的业务价值交付。当前实现中的体验损失项并非仅仅是“请求带宽减去实际带宽”的单一差值，而是由四部分组成：

$$
\Phi_{exp}
=
\frac{1}{|\mathcal{F}|}
\sum_{f\in\mathcal{F}}V_f
\left[
(1-z_f)
\frac{d_f^{ul}}{B_f^{ul}}
\frac{d_f^{dl}}{B_f^{dl}}
\frac{1}{2}\frac{\gamma_f^{ul}}{G_f^{ul}}
\frac{1}{2}\frac{\gamma_f^{dl}}{G_f^{dl}}
\right]
$$

其中：

- $(1-z_f)$ 表示流被完全拒绝；
- $d_f^{ul}, d_f^{dl}$ 表示上下行请求带宽未被满足；
- $\gamma_f^{ul}, \gamma_f^{dl}$ 表示上下行 GBR 未被满足。

因此，系统宁可牺牲大量低价值娱乐流量，也会尽力保护高价值控制流量。这个价值层级由 $V_f$ 在数学上显式体现。

### 4. 核心 QoS 违约项

当前实现将时延视为核心 QoS 约束，其违约项定义为：

$$
\Phi_{qos}^{core}
=
\frac{1}{|\mathcal{F}|}
\sum_{f\in\mathcal{F}}\sum_{j\in\mathcal{S}}
V_f q_{f,j}^{lat}x_{f,j}
$$

时延违约被纳入 $\omega_3$ 所控制的高权重项中，这意味着一条高优先级业务流即便在带宽上有解，只要被映射到高时延切片，也会在目标函数中遭受显著惩罚。

### 5. 辅助 QoS 违约项

丢包和抖动违约项被单独纳入 $\omega_4$ 控制：

$$
\Phi_{qos}^{aux}
=
\frac{1}{|\mathcal{F}|}
\sum_{f\in\mathcal{F}}\sum_{j\in\mathcal{S}}
V_f\left(q_{f,j}^{loss}+q_{f,j}^{jit}\right)x_{f,j}
$$

这是对模板中的另一处必要修正。你给出的模板将所有体验损失合并为单一项，但当前代码已经显式暴露了 $w_4$，因此将 loss/jitter 单独拆出更符合实际实现，也便于后续调参。

### 6. 平局打破项

为保证在多个完全等价解之间获得稳定输出，系统引入一个极小的平局打破项：

$$
\Phi_{tie}=\sum_{f\in\mathcal{F}}\sum_{j\in\mathcal{S}}\eta_j x_{f,j}
$$

其中 $\eta_j$ 为切片索引，$\varepsilon\ll 1$。该项不改变主优化方向，仅用于减小解的随机性。

## 三、模型解释与说明

综上，当前优化器并不是一个“所有流必须被完美满足”的刚性可行性检查器，而是一个面向在线资源冲突场景的、强调可落地性的混合资源分配模型。它回答的问题是：

> 在当前切片容量、节点算力和网络状态给定的条件下，如何以最小的全局代价完成业务接纳、降级、重映射与资源分摊？

与传统硬约束模型相比，当前模型的核心优势在于：

1. 即使资源紧张，仍然尽量保持问题可解；
2. 高优先级业务能够通过目标函数权重自然获得保护；
3. 增量优化与混合优化都具备工程上的稳定性；
4. 物理节点 CPU、PRB、MEC 约束与切片容量约束被统一纳入同一求解框架。

因此，该模型更适合作为 6G 多智能体意图编排系统中的在线切片决策内核，而不是单纯的离线理论分配器。
