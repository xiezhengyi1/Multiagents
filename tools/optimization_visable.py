import streamlit as st
import pandas as pd
import main as opt_engine
from typing import List

# 设置页面配置
st.set_page_config(
    page_title="5G切片资源优化平台",
    page_icon="📡",
    layout="wide"
)

# --- 状态管理 ---
if 'apps' not in st.session_state:
    apps, slices, nodes = opt_engine.get_initial_scenario()
    st.session_state['apps'] = apps
    st.session_state['slices'] = slices
    st.session_state['nodes'] = nodes
    st.session_state['optimization_results'] = None
    st.session_state['slice_stats'] = None

# --- 侧边栏配置 ---
st.sidebar.title("⚙️ 优化参数配置")
rho = st.sidebar.slider("目标负载率 (rho)", 0.1, 1.0, 0.8, 0.05)
w1 = st.sidebar.number_input("负载均衡权重 (w1)", value=100.0)
w2 = st.sidebar.number_input("信令开销权重 (w2)", value=50.0)
w3 = st.sidebar.number_input("体验损失权重 (w3)", value=1000.0)
alpha = st.sidebar.number_input("带宽转CPU系数 (alpha)", value=0.1)
beta = st.sidebar.number_input("带宽转内存系数 (beta)", value=0.05)

config = opt_engine.OptimizationConfig(rho=rho, w1=w1, w2=w2, w3=w3, alpha=alpha, beta=beta)
engine = opt_engine.SliceOptimizationEngine(config=config)

# --- 主界面 ---
st.title("📡 5G网络切片资源智能优化系统")

with st.expander("📖 优化问题数学模型定义"):
    st.markdown(r"""
    ## 一、 符号定义

    ### 1. 集合与索引

    - $A = \{a_1, a_2, \dots, a_N\}$：待接入的应用（App）集合。
    - $S = \{s_1, s_2, \dots, s_M\}$：现有的网络切片集合。
    - $K = \{k_1, k_2, \dots, k_L\}$：物理节点（Node）集合。
    - $Flows(s_j)$：当前已承载在切片 $s_j$ 上的所有 PDU 会话中的 QoS 流集合。

    ### 2. 决策变量

    - $x_{ij} \in \{0, 1\}$：**URSP 映射决策**。若将 App $a_i$ 的 URSP 规则指向切片 $s_j$，则为 1，否则为 0。

    ### 3. 常量参数

    - $Type(a_i)$：App $i$ 的业务类型需求（如 URLLC, eMBB）。
    - $SST(s_j)$：切片 $s_j$ 的标准切片类型。
    - $B_{if}, D_{if}, P_{if}$：App $i$ 中第 $f$ 个 QoS 流的带宽需求、时延上限、优先级。
    - $C_j^{net},C^{total}_j, D_j^{link}, D_j^{proc}$：切片 $j$ 的剩余可用容量、总共可用容量、当前链路测得时延、链路处理时延。
    - $R_k^{cpu}$：物理节点 $k$ 的剩余计算资源。

    ------

    ## 二、 约束条件

    ### 1. URSP 类型强制匹配

    PDU 会话分配的切片类型必须满足 App 的业务属性需求：

    $$x_{ij} = 1 \implies SST(s_j) = Type(a_i)$$

    ### 2. 时延约束

    时延是由链路物理特性和负载决定的，与带宽挤占逻辑不同。

    $$x_{ij} = 1 \implies (D_j^{link} + D_j^{proc}) \le D_{if}, \quad \forall f \in Flows(a_i)$$

    (只有当应用 $i$ 映射到切片 $j$ 时，该约束才生效)

    ### 3. 容量与挤占约束

    对于切片 $s_j$，其承载的新 App 带宽之和不得超过剩余容量。若超过，则检查优先级 $P_{i f}$：

    $$\sum_{i=1}^N B_{ij}^{act} \le C_j^{net} + \sum_{m \in Flows(s_j), P_m < P_i^{max}} B_m, \quad \forall j$$

       $0 \le B_{ij}^{act} \le x_{ij} \cdot (\sum_f B_{if})$

    > **说明**：如果资源不足且无法修改 URSP 映射，系统将释放切片内优先级低于当前请求的应用资源（挤占）。

    ### 4. 物理节点资源限制

    所有运行在物理节点 $k$ 上的切片资源总消耗（含新映射的应用）不能超过节点上限：

    $$\sum_{j \in S_k} \left( \text{CPU\_usage}_j + \sum_{i=1}^N x_{ij} \cdot \sum_f(\alpha B_{i f}) \right) \le R_k^{cpu}$$

    $$\sum_{j \in S_k} \left( \text{Memory\_usage}_j + \sum_{i=1}^N x_{ij}\sum_f(\beta B_{i f}) \right) \le R_k^{mem}$$

    ## 三、 目标函数

    采用多目标加权优化，旨在**负载均衡**和的同时**最小化策略变动开销**和**业务体验损失项**：

    $$\min Z = \omega_1 \underbrace{\sum_{j=1}^M \left( \frac{\sum_{i=1}^N B_{ij}^{act} + \text{Load}(s_j)}{C_j^{total}} - \rho \right)^2}_{\text{负载均衡项}} + \omega_2 \underbrace{\sum_{i=1}^N \sum_{j=1}^M |x_{ij} - x_{ij}^{old}|}_{\text{信令开销项}} + \omega_3 \underbrace{\sum_{i=1}^N V_i \left( \frac{B_i - \sum_{j=1}^M B_{ij}^{act}}{B_i} \right)}_{\text{业务体验损失项}}$$

    1. **负载均衡项**：使各切片负载趋于目标负载率 $\rho$，避免单点拥塞。（这一项是否需要）
    2. **信令开销项**：惩罚 URSP 规则的变动，对应方案 B 的成本。
    3. **业务体验损失项**：
       - 当执行**策略 C** 时，$B_{ij}^{act}$ 会小于 $B_i$，该项值增加，代表体验下降。
       - 当执行**策略 A**（拒绝/挤占）时，$\sum B_{ij}^{act} = 0$，体验损失达到最大（即 1）。
       - 通过权重 $V_i$，确保远程驾驶等关键业务即使只损失 1Mbps，其惩罚也远大于娱乐业务损失 10Mbps。
    """)

# 1. 当前系统状态展示
st.header("1. 当前系统状态")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 现有业务列表")
    app_data = []
    for app in st.session_state['apps']:
        # 格式化流信息
        flows_str = "; ".join([f"{f.name}({f.bw}M, {f.lat}ms, P{f.priority})" for f in app.flows])
        
        app_data.append({
            "App Name": app.name,
            "Type": app.type,
            "Total BW (Mbps)": app.total_bw,
            "Min Latency (ms)": app.min_lat,
            "Priority": app.max_prio,
            "Flows Details": flows_str,
            "Current Slice": app.old_slice if app.old_slice else "None"
        })
    st.dataframe(pd.DataFrame(app_data), width='stretch')

with col2:
    st.subheader("🍰 切片资源池")
    slice_data = []
    for s in st.session_state['slices']:
        slice_data.append({
            "Slice Name": s.name,
            "Type": s.sst,
            "Total Cap (Mbps)": s.total_bw,
            "Reserved (Mbps)": s.reserved_bw,
            "Latency (ms)": s.latency
        })
    st.dataframe(pd.DataFrame(slice_data), width='stretch')

    st.subheader("🖥️ 物理节点资源池")
    node_data = []
    for n in st.session_state['nodes']:
        # 计算节点当前负载
        current_cpu = 0
        current_mem = 0
        
        # 1. 切片基础负载
        for s_name in n.slices_hosted:
            s_obj = next((s for s in st.session_state['slices'] if s.name == s_name), None)
            if s_obj:
                current_cpu += s_obj.current_load_bw * alpha
                current_mem += s_obj.current_load_bw * beta
        
        # 2. 现有应用负载 (基于 old_slice)
        for app in st.session_state['apps']:
            if app.old_slice in n.slices_hosted:
                current_cpu += app.total_bw * alpha
                current_mem += app.total_bw * beta

        node_data.append({
            "Node Name": n.name,
            "CPU Cap": n.cpu_capacity,
            "Rem CPU": round(n.cpu_capacity - current_cpu, 2),
            "Mem Cap": n.memory_capacity,
            "Rem Mem": round(n.memory_capacity - current_mem, 2),
            "Hosted Slices": ", ".join(n.slices_hosted)
        })
    st.dataframe(pd.DataFrame(node_data), width='stretch')

# 2. 运行优化
st.header("2. 资源优化决策")

if st.button("🚀 运行全局优化", type="primary"):
    with st.spinner("正在计算最优切片映射策略..."):
        results_df, slice_stats_df = engine.solve(
            st.session_state['apps'], 
            st.session_state['slices'], 
            st.session_state['nodes']
        )
        st.session_state['optimization_results'] = results_df
        st.session_state['slice_stats'] = slice_stats_df
    st.success("优化完成！")

if st.session_state['optimization_results'] is not None:
    st.subheader("📊 优化结果")
    
    st.markdown("#### 📌 优化后的业务列表")
    
    # 高亮显示策略列
    def highlight_strategy(val):
        color = 'green' if '保持' in val else 'orange' if '重路由' in val else 'red'
        return f'color: {color}; font-weight: bold'
        
    st.dataframe(
        st.session_state['optimization_results'].style.map(highlight_strategy, subset=['Strategies']),
        width='stretch'
    )
    
    st.markdown("#### 🔋 切片负载率状态")
    st.dataframe(st.session_state['slice_stats'], width='stretch')
    
    # 简单的负载可视化
    st.bar_chart(
        st.session_state['slice_stats'].set_index("Slice")[["Allocated (M)", "Reserved (M)"]],
        stack=True
    )

# 3. 模拟新业务接入
st.header("3. 模拟新业务接入")
with st.expander("➕ 添加新业务请求", expanded=True):
    with st.form("new_app_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        new_name = col_f1.text_input("业务名称", value="Emergency_Call")
        new_type = col_f2.selectbox("业务类型", ["URLLC", "eMBB", "mMTC"])
        new_weight = col_f3.number_input("业务权重", value=5000)
        
        col_f4, col_f5, col_f6 = st.columns(3)
        flow_bw = col_f4.number_input("流带宽 (Mbps)", value=5.0)
        flow_lat = col_f5.number_input("时延要求 (ms)", value=5.0)
        flow_prio = col_f6.number_input("优先级", value=2000)
        
        submitted = st.form_submit_button("提交请求并重新优化")
        
        if submitted:
            # 创建新应用对象
            new_flow = opt_engine.Flow("Default_Flow", flow_bw, flow_lat, flow_prio)
            new_app = opt_engine.App(new_name, new_type, [new_flow], new_weight, old_slice=None)
            
            # 更新状态
            # 检查是否已存在同名应用，若存在则移除旧的（为了演示方便）
            st.session_state['apps'] = [a for a in st.session_state['apps'] if a.name != new_name]
            st.session_state['apps'].append(new_app)
            
            # 自动触发优化
            results_df, slice_stats_df = engine.solve(
                st.session_state['apps'], 
                st.session_state['slices'], 
                st.session_state['nodes']
            )
            st.session_state['optimization_results'] = results_df
            st.session_state['slice_stats'] = slice_stats_df
            
            st.rerun()

# 重置按钮
if st.sidebar.button("🔄 重置场景"):
    del st.session_state['apps']
    st.rerun()
