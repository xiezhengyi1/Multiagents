import streamlit as st
import pandas as pd
import json
import os
import sys

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.optimizer import optimize_network_slices, App, Slice, Node, Flow
from tools.db_tool import get_initial_scenario, cache_scenario, get_cached_scenario, deserialize_scenario_payload

# --- 页面配置 ---
st.set_page_config(
    page_title="6G切片优化器测试台",
    page_icon="🧪",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; color: #1E88E5; margin-bottom: 20px; }
    .section-header { font-size: 1.2rem; font-weight: bold; margin-top: 15px; margin-bottom: 10px; }
    .stButton>button { width: 100%; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- Session State 初始化 ---
if 'initialized' not in st.session_state:
    apps, slices, nodes = get_initial_scenario()
    st.session_state['apps'] = apps
    st.session_state['slices'] = slices
    st.session_state['nodes'] = nodes
    st.session_state['init_done'] = True
    st.session_state['last_result'] = None

# --- 侧边栏：配置 ---
with st.sidebar:
    st.title("⚙️ 求解器配置")
    
    st.markdown("### 1. 优化目标权重")
    w1 = st.slider("W1: 负载均衡 (Load)", 0.0, 500.0, 100.0)
    w2 = st.slider("W2: 迁移/信令 (Signal)", 0.0, 500.0, 50.0)
    w3 = st.slider("W3: 体验/带宽 (Exp)", 0.0, 2000.0, 1000.0)
    w4 = st.slider("W4: QoS (Loss/Jitter)", 0.0, 500.0, 100.0)
    
    st.markdown("### 2. 运行模式")
    opt_mode = st.radio(
        "选择优化模式:",
        ('full', 'incremental', 'hybrid'),
        format_func=lambda x: {
            'full': '全量优化 (Global)',
            'incremental': '严格增量 (Strict)',
            'hybrid': '混合/挤占 (Hybrid)'
        }[x],
        index=2
    )
    st.info({
        'full': "推倒重来：允许任意重路由，追求全局最优。",
        'incremental': "保守策略：固定旧流切片和带宽，仅优化新流。",
        'hybrid': "激进策略：固定旧流切片，但允许压缩旧流带宽以接纳高优先级新流。"
    }[opt_mode])

    if st.button("🔄 重置所有数据"):
        apps, slices, nodes = get_initial_scenario()
        st.session_state['apps'] = apps
        st.session_state['slices'] = slices
        st.session_state['nodes'] = nodes
        st.session_state['last_result'] = None
        st.rerun()

st.markdown('<div class="main-header">🧪 6G网络切片优化器 - 可视化测试台</div>', unsafe_allow_html=True)

# --- 第一部分：环境数据编辑 ---
col_slice, col_flow = st.columns([1, 1.2])

# 1. 切片列表编辑
with col_slice:
    st.markdown('<div class="section-header">1. 切片资源池 (Slice Resources)</div>', unsafe_allow_html=True)
    
    current_slices = st.session_state['slices']
    slice_dicts = []
    for s in current_slices:
        slice_dicts.append({
            "SNSSAI": s.snssai,
            "Name": s.name,
            "Cap_UL": s.total_bw_ul,
            "Cap_DL": s.total_bw_dl,
            "Reserved": s.reserved_bw,
            "Lat (ms)": s.latency,
            "Loss": s.loss,
            "Jitter": s.jitter
        })
    
    edited_slices = st.data_editor(
        pd.DataFrame(slice_dicts),
        key="editor_slices",
        num_rows="dynamic",
        column_config={
            "Lat (ms)": st.column_config.NumberColumn(min_value=0.0, format="%.1f"),
            "Loss": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, format="%.4f"),
            "Jitter": st.column_config.NumberColumn(min_value=0.0, format="%.1f"),
        },
        use_container_width=True
    )
    
    # 实时转换回对象
    new_slices_obj = []
    for _, row in edited_slices.iterrows():
        # 查找旧对象以保留某些状态 (current_load)，或者新建
        old_obj = next((s for s in current_slices if s.snssai == row['SNSSAI']), None)
        
        # 简单起见，从 SNSSAI 反推 SST/SD (如果未修改 SNSSAI)
        # 如果是新行，给个默认 SST/SD
        if old_obj:
            sst, sd = old_obj.sst, old_obj.sd
            cur_ul, cur_dl = old_obj.current_load_bw_ul, old_obj.current_load_bw_dl
        else:
            sst, sd = 1, "FFFFFF"
            cur_ul, cur_dl = 0.0, 0.0
            
        s_new = Slice(
            name=row['Name'],
            sst=sst, sd=sd, # 注意：data_editor没有暴露 SST/SD 编辑，简化展示
            total_bw_ul=row['Cap_UL'],
            total_bw_dl=row['Cap_DL'],
            current_load_bw_ul=cur_ul,
            current_load_bw_dl=cur_dl,
            latency=row['Lat (ms)'],
            proc_delay=1.0, 
            reserved_bw=row['Reserved'],
            loss=row['Loss'],
            jitter=row['Jitter']
        )
        new_slices_obj.append(s_new)
    st.session_state['slices'] = new_slices_obj

# 2. 现有业务流编辑 (Flattened View)
with col_flow:
    st.markdown('<div class="section-header">2. 现有业务流 (Active Flows)</div>', unsafe_allow_html=True)
    
    current_apps = st.session_state['apps']
    flow_flat_list = []
    for app in current_apps:
        for f in app.flows:
            flow_flat_list.append({
                "App_ID": app.app_id,
                "App_Name": app.name,
                "Flow_ID": f.flow_id,
                "Name": f.name,
                "BW_UL": f.bw_ul,
                "BW_DL": f.bw_dl,
                "GBR_UL": f.gbr_ul,
                "GBR_DL": f.gbr_dl,
                "Prio": f.priority,
                "Lat_Req": f.lat,
                "Loss_Req": f.loss_req,
                "Jitter_Req": f.jitter_req,
                "Slice": f.old_slice,
                "Act_UL": f.old_allocated_bw_ul,
                "Act_DL": f.old_allocated_bw_dl
            })
            
    edited_flows = st.data_editor(
        pd.DataFrame(flow_flat_list),
        key="editor_flows",
        num_rows="dynamic",
        column_config={
            "App_ID": st.column_config.TextColumn(disabled=False), # 允许修改归属App
            "Slice": st.column_config.TextColumn(disabled=True, help="由优化器分配"),
            "Act_UL": st.column_config.NumberColumn(disabled=True, format="%.2f"),
            "Act_DL": st.column_config.NumberColumn(disabled=True, format="%.2f"),
            "Loss_Req": st.column_config.NumberColumn(format="%.4f"),
        },
        use_container_width=True
    )
    
    # 重建 App 结构
    # Group by App_ID
    new_apps_dict = {}

    def params_check(val):
        return val is not None
    
    # 遍历编辑后的 DataFrame
    for _, row in edited_flows.iterrows():
        # 处理可能被用户删除行的情况
        # ...
        a_id = row['App_ID']
        a_name = row['App_Name']
        
        if params_check(a_id): # Simple validation
             if a_id not in new_apps_dict:
                new_apps_dict[a_id] = {
                    "name": a_name,
                    "flows": []
                }
            
             # 构造流
             f_obj = Flow(
                name=row['Name'],
                flow_id=row['Flow_ID'],
                bw_ul=row['BW_UL'],
                bw_dl=row['BW_DL'],
                gbr_ul=row['GBR_UL'],
                gbr_dl=row['GBR_DL'],
                lat=row['Lat_Req'],
                loss_req=row['Loss_Req'],
                jitter_req=row['Jitter_Req'],
                priority=int(row['Prio']),
                old_slice=row['Slice'] if pd.notna(row['Slice']) and row['Slice'] != "" else None,
                old_allocated_bw_ul=row['Act_UL'] if pd.notna(row['Act_UL']) else None,
                old_allocated_bw_dl=row['Act_DL'] if pd.notna(row['Act_DL']) else None
             )
             new_apps_dict[a_id]["flows"].append(f_obj)
        
    # Convert back to List[App]
    reconstructed_apps = []
    for a_id, data in new_apps_dict.items():
        reconstructed_apps.append(App(
            app_id=a_id,
            name=data['name'],
            flows=data['flows']
        ))
    st.session_state['apps'] = reconstructed_apps

# --- 第三部分：新业务请求 ---
st.markdown("---")
st.markdown('<div class="section-header">3. 发起新业务请求 (Request New Service)</div>', unsafe_allow_html=True)

with st.form("new_app_form"):
    c1, c2, c3 = st.columns(3)
    new_app_name = c1.text_input("App Name", "Emergency_Video")
    new_app_id = c2.text_input("App ID", "app_emerg_01")
    
    st.markdown("**Flow Parameters:**")
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    f_ul = fc1.number_input("UL BW (M)", 10.0)
    f_dl = fc2.number_input("DL BW (M)", 50.0)
    f_lat = fc3.number_input("Lat Req (ms)", 20.0)
    f_loss = fc4.number_input("Loss Req", 0.01, format="%.4f")
    f_jitter = fc5.number_input("Jitter Req (ms)", 10.0)
    
    fc6, fc7, fc8 = st.columns(3)
    f_prio = fc6.number_input("Priority (1=High)", 1, 100, 1)
    f_gbr_ul = fc7.number_input("GBR UL (M)", 2.0)
    f_gbr_dl = fc8.number_input("GBR DL (M)", 10.0)
    
    submitted = st.form_submit_button("🚀 提交并在当前上下文中优化 (Run Optimizer)")

if submitted:
    # 构造请求字典
    req_data = {
        "name": new_app_name,
        "app_id": new_app_id,
        "flows": [{
            "name": "Default_Stream",
            "flow_id": "f_main",
            "bw_ul": f_ul,
            "bw_dl": f_dl,
            "gbr_ul": f_gbr_ul,
            "gbr_dl": f_gbr_dl,
            "lat": f_lat,
            "loss_req": f_loss,
            "jitter_req": f_jitter,
            "priority": int(f_prio)
        }]
    }

    # 同步环境数据到缓存（供优化器读取）
    cache_scenario(
        st.session_state['apps'],
        st.session_state['slices'],
        st.session_state['nodes']
    )
    
    with st.spinner(f"Running Optimization (Mode={opt_mode})..."):
        report = optimize_network_slices(
            req_data,
            w1=w1, w2=w2, w3=w3, w4=w4,
            mode=opt_mode
        )
        st.session_state['last_result'] = report

        if isinstance(report, dict) and report.get("scenario"):
            parsed = deserialize_scenario_payload(report)
            if parsed:
                apps, slices, nodes = parsed
                st.session_state['apps'] = apps
                st.session_state['slices'] = slices
                st.session_state['nodes'] = nodes

# --- 结果展示 ---
if st.session_state['last_result']:
    st.markdown("---")
    st.markdown('<div class="section-header">4. 优化结果报告</div>', unsafe_allow_html=True)
    st.text_area("Result Log", st.session_state['last_result'], height=400)
    
    # 简单的可视化：切片负载对比
    st.markdown("##### 切片负载概览 (Slice Load)")
    
    # 从缓存取最新切片状态
    cached_apps, cached_slices, _ = get_cached_scenario()
    latest_slices = cached_slices if cached_slices else st.session_state['slices']
    
    load_data = []
    # 重新计算负载用于展示 (简单累加)
    latest_apps = cached_apps if cached_apps else st.session_state['apps']
    
    for s in latest_slices:
        used_ul = s.reserved_bw + s.current_load_bw_ul
        used_dl = s.reserved_bw + s.current_load_bw_dl
        
        # 加上 Apps 的负载
        for app in latest_apps:
            for f in app.flows:
                if f.old_slice == s.snssai: # 此时 old_slice 已经被更新为最新分配结果
                    used_ul += (f.old_allocated_bw_ul or 0)
                    used_dl += (f.old_allocated_bw_dl or 0)

        load_data.append({
            "Slice": s.name,
            "Used UL": used_ul,
            "Total UL": s.total_bw_ul,
            "Used DL": used_dl,
            "Total DL": s.total_bw_dl,
        })
    
    df_load = pd.DataFrame(load_data)
    if not df_load.empty:
        st.bar_chart(df_load.set_index("Slice")[["Used UL", "Used DL"]])

