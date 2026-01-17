import streamlit as st
import pandas as pd
import os
import sys
from typing import List

# 添加项目根目录到 sys.path，确保能够导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimizer import (
    get_initial_scenario,
    SliceOptimizationEngine,
    OptimizationConfig,
    App,
    Flow,
    Slice,
    Node,
)

st.set_page_config(
    page_title="优化器可视化测试",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.0rem;
            color: #1E88E5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def _to_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)

def _to_int(value, default=0):
    try:
        if pd.isna(value):
            return int(default)
        return int(value)
    except Exception:
        return int(default)

def _to_str(value, default=""):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value)

def _slices_to_df(slices: List[Slice]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "name": s.name,
            "sst": s.sst,
            "sd": s.sd,
            "total_bw_ul": s.total_bw_ul,
            "total_bw_dl": s.total_bw_dl,
            "current_load_bw_ul": s.current_load_bw_ul,
            "current_load_bw_dl": s.current_load_bw_dl,
            "latency": s.latency,
            "proc_delay": s.proc_delay,
            "loss": s.loss,
            "jitter": s.jitter,
            "reserved_bw": s.reserved_bw,
        }
        for s in slices
    ])

def _flows_to_df(apps: List[App]) -> pd.DataFrame:
    rows = []
    for app in apps:
        for f in app.flows:
            rows.append({
                "app_id": app.app_id,
                "app_name": app.name,
                "flow_id": f.flow_id,
                "name": f.name,
                "bw_ul": f.bw_ul,
                "bw_dl": f.bw_dl,
                "gbr_ul": f.gbr_ul,
                "gbr_dl": f.gbr_dl,
                "lat": f.lat,
                "loss_req": f.loss_req,
                "jitter_req": f.jitter_req,
                "priority": f.priority,
                "old_slice": f.old_slice,
                "old_allocated_bw_ul": f.old_allocated_bw_ul,
                "old_allocated_bw_dl": f.old_allocated_bw_dl,
            })
    return pd.DataFrame(rows)

def _build_slices(df: pd.DataFrame) -> List[Slice]:
    slices = []
    for _, row in df.iterrows():
        name = _to_str(row.get("name"), "Slice")
        sst = _to_int(row.get("sst"), 1)
        sd = _to_str(row.get("sd"), "000001").upper()
        slices.append(
            Slice(
                name=name,
                sst=sst,
                sd=sd,
                total_bw_ul=_to_float(row.get("total_bw_ul"), 0),
                total_bw_dl=_to_float(row.get("total_bw_dl"), 0),
                current_load_bw_ul=_to_float(row.get("current_load_bw_ul"), 0),
                current_load_bw_dl=_to_float(row.get("current_load_bw_dl"), 0),
                latency=_to_float(row.get("latency"), 0),
                proc_delay=_to_float(row.get("proc_delay"), 0),
                loss=_to_float(row.get("loss"), 0),
                jitter=_to_float(row.get("jitter"), 0),
                reserved_bw=_to_float(row.get("reserved_bw"), 0),
            )
        )
    return slices

def _build_apps(df: pd.DataFrame) -> List[App]:
    apps = []
    if df.empty:
        return apps

    for (app_id, app_name), group in df.groupby(["app_id", "app_name"], dropna=False):
        flows = []
        for idx, row in group.iterrows():
            flow_id = _to_str(row.get("flow_id"), f"{app_id}_f{idx}")
            name = _to_str(row.get("name"), flow_id)
            bw_ul = _to_float(row.get("bw_ul"), 0)
            bw_dl = _to_float(row.get("bw_dl"), 0)
            gbr_ul = _to_float(row.get("gbr_ul"), bw_ul)
            gbr_dl = _to_float(row.get("gbr_dl"), bw_dl)
            lat = _to_float(row.get("lat"), 100)
            loss_req = _to_float(row.get("loss_req"), 0.05)
            jitter_req = _to_float(row.get("jitter_req"), 50)
            priority = _to_int(row.get("priority"), 10)
            old_slice = _to_str(row.get("old_slice"), None)
            old_allocated_bw_ul = _to_float(row.get("old_allocated_bw_ul"), bw_ul)
            old_allocated_bw_dl = _to_float(row.get("old_allocated_bw_dl"), bw_dl)
            flows.append(
                Flow(
                    name=name,
                    flow_id=flow_id,
                    bw_ul=bw_ul,
                    bw_dl=bw_dl,
                    gbr_ul=gbr_ul,
                    gbr_dl=gbr_dl,
                    lat=lat,
                    loss_req=loss_req,
                    jitter_req=jitter_req,
                    priority=priority,
                    old_slice=old_slice if old_slice else None,
                    old_allocated_bw_ul=old_allocated_bw_ul,
                    old_allocated_bw_dl=old_allocated_bw_dl,
                )
            )
        apps.append(App(name=_to_str(app_name, app_id), app_id=_to_str(app_id, "app"), flows=flows))
    return apps

def _ensure_nodes_cover_slices(nodes: List[Node], slices: List[Slice]) -> List[Node]:
    slice_names = [s.name for s in slices]
    if not nodes:
        return [Node("Node_All", cpu_capacity=1_000_000, memory_capacity=1_000_000, slices_hosted=slice_names)]
    covered = set()
    for n in nodes:
        for s_name in n.slices_hosted:
            if s_name in slice_names:
                covered.add(s_name)
    missing = [s_name for s_name in slice_names if s_name not in covered]
    if missing:
        nodes[0].slices_hosted = list(dict.fromkeys(nodes[0].slices_hosted + missing))
    return nodes

if "initialized" not in st.session_state:
    apps, slices, nodes = get_initial_scenario()
    st.session_state["apps"] = apps
    st.session_state["slices"] = slices
    st.session_state["nodes"] = nodes
    st.session_state["flow_results"] = None
    st.session_state["slice_results"] = None
    st.session_state["solve_status"] = None
    st.session_state["objective_value"] = None
    st.session_state["objective_breakdown"] = None
    st.session_state["initialized"] = True

st.markdown('<div class="main-header">📊 优化器可视化测试界面</div>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.subheader("参数设置")
    w1 = st.number_input("w1 负载均衡", min_value=0.0, value=100.0, step=1.0)
    w2 = st.number_input("w2 信令开销", min_value=0.0, value=50.0, step=1.0)
    w3 = st.number_input("w3 体验损失", min_value=0.0, value=1000.0, step=10.0)
    incremental_mode = st.checkbox("启用增量优化", value=False)

    if st.button("重置为默认场景"):
        apps, slices, nodes = get_initial_scenario()
        st.session_state["apps"] = apps
        st.session_state["slices"] = slices
        st.session_state["nodes"] = nodes
        st.session_state["flow_results"] = None
        st.session_state["slice_results"] = None
        st.session_state["solve_status"] = None
        st.session_state["objective_value"] = None
        st.session_state["objective_breakdown"] = None
        st.rerun()

left, right = st.columns([1, 1])

with left:
    st.subheader("原切片列表")
    slice_df = _slices_to_df(st.session_state["slices"])
    edited_slices = st.data_editor(
        slice_df,
        num_rows="dynamic",
        key="slice_editor",
        column_config={
            "name": st.column_config.TextColumn("切片名称"),
            "sst": st.column_config.NumberColumn("SST", min_value=0, max_value=255, step=1),
            "sd": st.column_config.TextColumn("SD (Hex)", max_chars=6),
            "total_bw_ul": st.column_config.NumberColumn("总带宽 UL"),
            "total_bw_dl": st.column_config.NumberColumn("总带宽 DL"),
            "current_load_bw_ul": st.column_config.NumberColumn("当前负载 UL"),
            "current_load_bw_dl": st.column_config.NumberColumn("当前负载 DL"),
            "latency": st.column_config.NumberColumn("链路时延"),
            "proc_delay": st.column_config.NumberColumn("处理时延"),
            "loss": st.column_config.NumberColumn("丢包率(0~1)", min_value=0.0, max_value=1.0, step=0.001),
            "jitter": st.column_config.NumberColumn("抖动(ms)", min_value=0.0),
            "reserved_bw": st.column_config.NumberColumn("保留带宽"),
        },
        use_container_width=True,
    )

    preview_slices = _build_slices(edited_slices)
    if preview_slices:
        st.caption("S-NSSAI 预览")
        st.dataframe(
            pd.DataFrame([
                {"name": s.name, "snssai": s.snssai, "sst": s.sst, "sd": s.sd}
                for s in preview_slices
            ]),
            use_container_width=True,
        )

with right:
    st.subheader("原业务表（Flow 级）")
    flow_df = _flows_to_df(st.session_state["apps"])
    slice_options = [s.snssai for s in _build_slices(edited_slices)] if not edited_slices.empty else []
    edited_flows = st.data_editor(
        flow_df,
        num_rows="dynamic",
        key="flow_editor",
        column_config={
            "app_id": st.column_config.TextColumn("应用ID"),
            "app_name": st.column_config.TextColumn("应用名称"),
            "flow_id": st.column_config.TextColumn("流ID"),
            "name": st.column_config.TextColumn("流名称"),
            "bw_ul": st.column_config.NumberColumn("BW UL"),
            "bw_dl": st.column_config.NumberColumn("BW DL"),
            "gbr_ul": st.column_config.NumberColumn("GBR UL"),
            "gbr_dl": st.column_config.NumberColumn("GBR DL"),
            "lat": st.column_config.NumberColumn("时延(ms)"),
            "loss_req": st.column_config.NumberColumn("丢包率上限(0~1)", min_value=0.0, max_value=1.0, step=0.001),
            "jitter_req": st.column_config.NumberColumn("抖动上限(ms)", min_value=0.0),
            "priority": st.column_config.NumberColumn("优先级"),
            "old_slice": st.column_config.SelectboxColumn("原切片(S-NSSAI)", options=[""] + slice_options),
            "old_allocated_bw_ul": st.column_config.NumberColumn("原分配 UL"),
            "old_allocated_bw_dl": st.column_config.NumberColumn("原分配 DL"),
        },
        use_container_width=True,
    )

apply_col, run_col = st.columns([1, 1])
if apply_col.button("应用修改"):
    st.session_state["slices"] = _build_slices(edited_slices)
    st.session_state["apps"] = _build_apps(edited_flows)
    st.session_state["nodes"] = _ensure_nodes_cover_slices(st.session_state["nodes"], st.session_state["slices"])
    st.session_state["flow_results"] = None
    st.session_state["slice_results"] = None
    st.session_state["solve_status"] = None
    st.session_state["objective_value"] = None
    st.session_state["objective_breakdown"] = None
    st.success("已应用修改")

if run_col.button("运行优化", type="primary"):
    if not edited_slices.empty and not edited_flows.empty:
        st.session_state["slices"] = _build_slices(edited_slices)
        st.session_state["apps"] = _build_apps(edited_flows)
        st.session_state["nodes"] = _ensure_nodes_cover_slices(st.session_state["nodes"], st.session_state["slices"])
    if not st.session_state["apps"] or not st.session_state["slices"]:
        st.error("请先配置切片与业务流。")
    else:
        engine = SliceOptimizationEngine(OptimizationConfig(w1=w1, w2=w2, w3=w3))
        if incremental_mode:
            flow_results, slice_results, status_str, objective_val, breakdown = engine.solve_incremental(
                st.session_state["apps"],
                st.session_state["slices"],
                st.session_state["nodes"]
            )
        else:
            flow_results, slice_results, status_str, objective_val, breakdown = engine.solve(
                st.session_state["apps"],
                st.session_state["slices"],
                st.session_state["nodes"]
            )
        st.session_state["flow_results"] = flow_results
        st.session_state["slice_results"] = slice_results
        st.session_state["solve_status"] = status_str
        st.session_state["objective_value"] = objective_val
        st.session_state["objective_breakdown"] = breakdown
        st.success("求解完成")

st.markdown("---")
st.subheader("结果查看")

status_col, obj_col = st.columns([1, 2])
with status_col:
    status_text = st.session_state.get("solve_status")
    st.metric("求解状态", status_text if status_text else "-" )
with obj_col:
    obj_val = st.session_state.get("objective_value")
    st.metric("目标函数值", f"{obj_val:.6f}" if obj_val is not None else "-")

breakdown = st.session_state.get("objective_breakdown")
if breakdown:
    with st.expander("目标项分解"):
        st.write(
            {
                "load_norm": breakdown.get("load_norm"),
                "signal_norm": breakdown.get("signal_norm"),
                "exp": breakdown.get("exp"),
                "qos_norm": breakdown.get("qos_norm"),
                "tiebreak": breakdown.get("tiebreak"),
            }
        )

if st.session_state.get("flow_results") is not None:
    st.markdown("**Flow 分配结果**")
    st.dataframe(st.session_state["flow_results"], use_container_width=True)
else:
    st.info("暂无结果，请先运行优化。")

if st.session_state.get("slice_results") is not None:
    st.markdown("**切片负载状态**")
    st.dataframe(st.session_state["slice_results"], use_container_width=True)

