import streamlit as st
import pandas as pd
import json
import os
import sys

# 添加项目根目录到 sys.path，确保能够导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools import optimization as opt_engine
from tools.optimization import App, Slice, Node, Flow, set_global_scenario
from multi_agents import IntentEncodingAgent, OptimizationStrategyAgent
from utils.logger import setup_logger # 虽然前端不一定看控制台，但保持一致性

# --- 页面配置 ---
st.set_page_config(
    page_title="6G网络体验保障多智能体编排系统",
    page_icon="🤖",
    layout="wide"
)

# --- 样式优化 ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State 初始化 ---
if 'initialized' not in st.session_state:
    apps, slices, nodes = opt_engine.get_initial_scenario()
    st.session_state['apps'] = apps
    st.session_state['slices'] = slices
    st.session_state['nodes'] = nodes
    st.session_state['logs'] = []
    st.session_state['initialized'] = True
else:
    # 热修复：检查 Slice 是否包含新加的 'sd' 属性，如果没有则重置
    if st.session_state.get('slices') and not hasattr(st.session_state['slices'][0], 'sd'):
        opt_engine = sys.modules.get('tools.optimization', opt_engine) # 尝试获取最新模块
        # 注意：这里可能需要重新 reload 模块，但在 Streamlit 中通常由 watcher 处理。
        # 简单起见，直接调用当前的 opt_engine.get_initial_scenario
        apps, slices, nodes = opt_engine.get_initial_scenario()
        st.session_state['apps'] = apps
        st.session_state['slices'] = slices
        st.session_state['nodes'] = nodes
        st.toast("系统已自动更新内部数据结构以匹配新代码。", icon="🔄")

if 'agent_results' not in st.session_state:
    st.session_state['agent_results'] = None

if 'final_report' not in st.session_state:
    st.session_state['final_report'] = None

# --- 侧边栏：环境配置 ---
with st.sidebar:
    st.title("🛠️ 环境配置")
    
    # API 配置
    with st.expander("🔑 API Key 设置", expanded=False):
        api_key = st.text_input("OpenAI/DashScope API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
        base_url = st.text_input("Base URL", value=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
    
    # 功能区：管理切片
    st.subheader("1. 切片资源池管理")
    
    # 将切片对象转换为 DataFrame 供 data_editor 使用
    slice_data = []
    for s in st.session_state['slices']:
        slice_data.append({
            "name": s.name,
            "sst": s.sst,
            "sd": s.sd,
            "total_bw": s.total_bw,
            "reserved_bw": s.reserved_bw,
            "latency": s.latency
        })
    df_slices = pd.DataFrame(slice_data)
    
    edited_slices_df = st.data_editor(
        df_slices, 
        num_rows="dynamic", 
        key="slice_editor",
        column_config={
            "name": "切片名称",
            "sst": st.column_config.NumberColumn("SST (1:eMBB,2:URLLC,3:MIoT)", min_value=0, max_value=255, step=1),
            "sd": st.column_config.TextColumn("SD (Hex)", max_chars=6, validate="^[0-9A-Fa-f]{6}$"),
            "total_bw": st.column_config.NumberColumn("总带宽(Mbps)", min_value=0),
            "reserved_bw": st.column_config.NumberColumn("保留带宽(Mbps)", min_value=0),
            "latency": st.column_config.NumberColumn("基础时延(ms)", min_value=0)
        }
    )
    
    # 功能区：管理现有业务
    st.subheader("2. 现有背景业务管理")
    app_display_data = []
    for app in st.session_state['apps']:
         app_display_data.append({
             "name": app.name,
             "total_bw": app.total_bw,
             "priority": app.max_prio,
             "slice": app.old_slice
         })
    df_apps = pd.DataFrame(app_display_data)
    
    edited_apps_df = st.data_editor(
        df_apps,
        num_rows="dynamic",
        key="app_editor",
         column_config={
            "name": "应用名称",
            "total_bw": st.column_config.NumberColumn("总带宽", disabled=True, help="由流汇总计算"),
         }
    )

    if st.button("🔄 重置/应用配置更改"):
        # 1. 更新切片对象
        new_slices = []
        for index, row in edited_slices_df.iterrows():
            # 尝试查找现有对象以保留不需要更改的属性(如 current_load)，或者直接新建
            # 这里简化处理：直接新建，这会重置 current_load，但在下一次 solve 时 current_load 是根据 app 分配计算的
            # 不过 wait, tools.optimization.Slice 中 current_load_bw 是其中一个字段。
            # 如果我们新建 Slice，current_load_bw 初始设为多少？
            # 原始代码中 get_initial_scenario 给定了一些初始值。
            # 既然是"自由配置"，我们假设现有背景业务的分配会决定 load。
            # 因此这里 current_load_bw 初始设为 0，依靠优化引擎重新计算背景业务的负载。
            
            s = Slice(
                name=row['name'],
                sst=int(row['sst']),
                sd=str(row['sd']),
                total_bw=float(row['total_bw']),
                current_load_bw=0.0, # 重新计算
                latency=float(row['latency']),
                proc_delay=1.0, # 默认值
                reserved_bw=float(row['reserved_bw'])
            )
            new_slices.append(s)
        st.session_state['slices'] = new_slices
        
        # 2. 更新应用对象 (较复杂，因为 App 包含 Flow)
        # 简易版：只允许删除或修改顶层属性。如果要修改流，这在 data_editor 里不好展示嵌套结构。
        # 我们保留原始 session_state['apps'] 中未被删除的项。
        current_app_names = edited_apps_df['name'].tolist()
        new_apps_list = []
        for app in st.session_state['apps']:
            if app.name in current_app_names:
                new_apps_list.append(app)
        st.session_state['apps'] = new_apps_list
        
        st.success("配置已更新！背景业务负载将在下次优化时重新计算。")

# --- 主界面 ---
st.markdown('<div class="main-header">🤖 6G网络切片多智能体编排系统</div>', unsafe_allow_html=True)
st.markdown("---")

# 1. 网络状态看板
st.subheader("📊 当前网络状态视图")

col1, col2 = st.columns([2, 1])

with col1:
    # 计算当前负载（简单模拟，基于现有 App 的 old_slice）
    slice_status = []
    for s in st.session_state['slices']:
        used_bw = s.reserved_bw
        for a in st.session_state['apps']:
            if a.old_slice == s.name:
                used_bw += a.total_bw
        
        slice_status.append({
            "切片名称": s.name,
            "总容量": s.total_bw,
            "已用(含保留)": used_bw,
            "使用率": min(100, (used_bw / s.total_bw) * 100) if s.total_bw > 0 else 0,
            "剩余": max(0, s.total_bw - used_bw)
        })
    df_status = pd.DataFrame(slice_status)
    st.dataframe(df_status)

with col2:
    st.info("💡 说明：\n- 左侧边栏可以自由增删切片和业务。\n- 下方输入意图，智能体将自动分析并调用求解器。")

# --- 业务流详细展示 ---
st.subheader("📋 业务流详细资源分布")
with st.expander("展开查看每个业务流的带宽与切片分配详情", expanded=True):
    all_flows_data = []
    if 'apps' in st.session_state and st.session_state['apps']:
        for app in st.session_state['apps']:
            assigned_slice = app.old_slice if app.old_slice else "未分配"
            # 兼容 app.flows 可能为空的情况
            if hasattr(app, 'flows') and app.flows:
                for f in app.flows:
                    all_flows_data.append({
                        "所属业务": app.name,
                        "流名称": f.name,
                        "占用带宽 (Mbps)": f.bw,
                        "时延要求 (ms)": f.lat,
                        "优先级": f.priority,
                        "分配切片": assigned_slice
                    })
            else:
                 # 如果没有流详情，仅显示业务本体
                 all_flows_data.append({
                    "所属业务": app.name,
                    "流名称": "Total (No sub-flows)",
                    "占用带宽 (Mbps)": app.total_bw,
                    "时延要求 (ms)": app.min_lat,
                    "优先级": app.max_prio,
                    "分配切片": assigned_slice
                })

        if all_flows_data:
            st.dataframe(pd.DataFrame(all_flows_data))
        else:
            st.info("暂无业务流数据")

# 2. 智能体交互区
st.markdown("---")
st.subheader("🗣️ 用户意图输入")

col_intent, col_context = st.columns([1, 1])

with col_intent:
    st.markdown("##### 1. UserIntent (自然语言)")
    user_input = st.text_area(
        "请输入您的接入需求:", 
        value="我是应急抢险指挥车(App_4)，需要接入网络。\n业务流1：远程机械臂控制，带宽20Mbps，不仅要大带宽而且时延不能超过10ms，这是关键控制流！\n业务流2：现场多路4K高清直播，带宽50Mbps，时延60ms左右。",
        height=300
    )

with col_context:
    st.markdown("##### 2. UeSmDecisionContext (结构化上下文)")
    with st.container(border=True):
        st.caption("UE & Session Context")
        c1, c2 = st.columns(2)
        with c1:
            supi = st.text_input("SUPI", value="imsi-46000000001")
            pdu_id = st.number_input("PDU Session ID", min_value=1, value=1)
            dnn = st.text_input("DNN", value="default")
        with c2:
            rat_type = st.selectbox("RAT Type", ["NR", "EUTRA"], index=0)
            access_type = st.selectbox("Access Type", ["3GPP_ACCESS", "NON_3GPP_ACCESS"], index=0)
            pdu_type = st.selectbox("PDU Type", ["IPV4", "IPV6", "ETHERNET"], index=0)
        
        st.caption("Slice Info (S-NSSAI)")
        c3, c4 = st.columns(2)
        with c3:
            sst = st.number_input("SST", min_value=0, max_value=255, value=1)
        with c4:
            sd = st.text_input("SD (Hex)", value="000000")

        # 构建结构化上下文对象
        ue_context_data = {
            "supi": supi,
            "am_info": {
                "user_location": None,
                "serving_plmn": None,
                "access_type": access_type,
                "rat_type": rat_type
            },
            "target_session": {
                "policy_context": {
                    "supi": supi,
                    "pduSessionId": pdu_id,
                    "sliceInfo": {"sst": sst, "sd": sd},
                    "dnn": dnn,
                    "pduSessionType": pdu_type,
                    "accessType": access_type,
                    "ratType": rat_type
                },
                "subscription_data": {
                    "content": {}
                },
                "remain_gbr_ul": None,
                "remain_gbr_dl": None,
                "active_app_session_ids": [],
                "traffic_influence_data": {}
            }
        }

col_run, col_reset = st.columns([1, 6])
run_btn = col_run.button("🚀 开始编排", type="primary")

if run_btn:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("请先在左侧边栏配置 API Key！")
    else:
        # !!! 关键步骤：将当前的前端配置注入到后端工具的上下文中 !!!
        set_global_scenario(
            st.session_state['apps'],
            st.session_state['slices'],
            st.session_state['nodes']
        )
        
        status_text = st.empty()
        result_container = st.container()

        try:
            # Step 1: 意图分析
            status_text.markdown("#### 🔄 Step 1: Intent Encoding Agent 正在分析意图...")
            intent_agent = IntentEncodingAgent(model_name="qwen-plus") # 使用更强的模型
            
            # 将结构化上下文转为 JSON 字符串传递给 Agent
            context_str = json.dumps(ue_context_data, indent=2, ensure_ascii=False)
            user_intent = intent_agent.analyze_intent(user_input, context=context_str)
            
            if user_intent:
                with result_container:
                    with st.expander("🧠 意图识别结果", expanded=False):
                        st.json(user_intent.dict())
            else:
                st.error("意图识别失败。")
                st.stop()
            
            # Step 2: 策略制定与执行
            status_text.markdown("#### 🔄 Step 2: Optimization Strategy Agent 正在制定策略并调用求解器...")
            
            # 构造网络状态描述供 LLM 参考
            # network_desc = "当前切片资源状态:\n"
            # for _, row in df_status.iterrows():
            #    network_desc += f"- {row['切片名称']}: 总容量 {row['总容量']}M, 剩余可用 {row['剩余']}M.\n"
            
            strategy_agent = OptimizationStrategyAgent(model_name="qwen-plus")
            final_report = strategy_agent.generate_strategy(user_intent.dict())
            
            # 保存结果到 Session State
            st.session_state['final_report'] = final_report
            
            status_text.success("✅ 编排完成！正在刷新全网状态视图...")
            
            # 强制刷新页面以更新顶部的表格
            import time
            time.sleep(1.5) # 给用户一点时间看到成功提示
            st.rerun()
            
        except Exception as e:
            st.error(f"运行过程中发生错误: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

# 3. 结果展示区 (如果有历史结果)
if st.session_state['final_report']:
    st.markdown("---")
    st.markdown("### 📝 最终执行报告")
    
    report_data = st.session_state['final_report']
    
    # 检查是否为结构化对象 (OutputStrategy)
    if hasattr(report_data, 'policy_type'):
        with st.container(border=True):
            st.markdown(f"**策略类型:** `{report_data.policy_type}`")
            st.markdown("**推荐动作:**")
            for action in report_data.recommended_actions:
                st.markdown(f"- {action}")
            
            st.markdown("---")
            st.markdown("**策略详细内容 (Policy Details):**")
            # 兼容 Pydantic v1 vs v2
            details_dict = report_data.policy_details.dict() if hasattr(report_data.policy_details, 'dict') else report_data.policy_details
            st.json(details_dict)
            
    else:
        # 兼容旧版本字符串
        st.markdown(f"""
        <div class="card">
            {str(report_data).replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🗑️ 清除报告并重置视图"):
        st.session_state['final_report'] = None
        st.rerun()

