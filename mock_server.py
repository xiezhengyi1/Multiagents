import time
import random
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from typing import Any, Dict, List, Optional, Tuple

# 配置
HOST = 'localhost'
PORT = 8000

# policy_id -> 最近一次RAN模拟结果摘要
RAN_ALLOCATION_CACHE: Dict[str, Dict[str, Any]] = {}


def _flatten_flows_from_apps(apps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从 snapshot/apps 中提取 flow 列表。"""
    flows: List[Dict[str, Any]] = []
    for app in apps:
        app_flows = app.get("flows", []) if isinstance(app, dict) else []
        for flow in app_flows:
            if isinstance(flow, dict):
                flows.append(flow)
    return flows


def _normalize_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """为 JointSimulator 补齐节点关键字段。"""
    out: List[Dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        item = dict(node)
        item.setdefault("name", f"Node_{idx}")
        item.setdefault("id", idx)
        item.setdefault("type", "Generic")
        item.setdefault("cpu_capacity", 100.0)
        item.setdefault("memory_capacity", 256.0)
        item.setdefault("mec_capacity", 20.0)
        item.setdefault("prb_capacity", 135.0)
        item.setdefault("slices_hosted", [])
        out.append(item)
    return out


def _default_service_types(flows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """当输入缺失 service_types 时，按 flow.service_type_id 生成默认服务类型。"""
    service_ids = sorted({int(f.get("service_type_id", 1)) for f in flows if isinstance(f, dict)})
    if not service_ids:
        service_ids = [1]
    return [
        {
            "id": sid,
            "name": f"service_{sid}",
            "critical_kpis": [1, 2, 3],
        }
        for sid in service_ids
    ]


def _extract_input_from_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """从请求体提取联合仿真输入。"""
    if not isinstance(payload, dict):
        return {}

    flows_data = payload.get("flows")
    if not flows_data and isinstance(payload.get("apps"), list):
        flows_data = _flatten_flows_from_apps(payload.get("apps", []))

    return {
        "flows_data": flows_data if isinstance(flows_data, list) else [],
        "nodes_data": payload.get("nodes") if isinstance(payload.get("nodes"), list) else [],
        "service_types_data": payload.get("service_types") if isinstance(payload.get("service_types"), list) else [],
        "sla_profiles_data": payload.get("sla_profiles") if isinstance(payload.get("sla_profiles"), list) else [],
    }


def _extract_input_from_snapshot(snapshot_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """从 DB 快照中提取联合仿真输入。"""
    if not isinstance(snapshot_data, dict):
        return {}

    apps = snapshot_data.get("apps", [])
    flows_data = _flatten_flows_from_apps(apps) if isinstance(apps, list) else []

    return {
        "flows_data": flows_data,
        "nodes_data": snapshot_data.get("nodes") if isinstance(snapshot_data.get("nodes"), list) else [],
        "service_types_data": snapshot_data.get("service_types") if isinstance(snapshot_data.get("service_types"), list) else [],
        "sla_profiles_data": snapshot_data.get("sla_profiles") if isinstance(snapshot_data.get("sla_profiles"), list) else [],
    }


def _merge_request_with_db(
    request_input: Dict[str, Any],
    db_input: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """请求体优先，数据库兜底。"""
    merged: Dict[str, Any] = {}
    source_used = set()

    for key in ("flows_data", "nodes_data", "service_types_data", "sla_profiles_data"):
        req_val = request_input.get(key, [])
        db_val = db_input.get(key, [])

        if req_val:
            merged[key] = req_val
            source_used.add("request")
        elif db_val:
            merged[key] = db_val
            source_used.add("db")
        else:
            merged[key] = []

    if source_used == {"request"}:
        source = "request"
    elif source_used == {"db"}:
        source = "db"
    elif source_used == {"request", "db"}:
        source = "mixed"
    else:
        source = "none"

    return merged, source


def _prepare_joint_inputs(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """准备 JointSimulator.run 需要的输入。"""
    request_input = _extract_input_from_request(payload)

    # 关键步骤: 请求体优先，缺失项从 DB 快照兜底
    db_input: Dict[str, Any] = {}
    try:
        from tools.db_tool import get_latest_snapshot_data
        snapshot_data = get_latest_snapshot_data()
        db_input = _extract_input_from_snapshot(snapshot_data)
    except Exception as exc:
        print(f"[MockServer] DB快照读取失败: {exc}")

    merged, source = _merge_request_with_db(request_input, db_input)

    flows = merged.get("flows_data", [])
    nodes = _normalize_nodes(merged.get("nodes_data", []))
    service_types = merged.get("service_types_data", [])
    sla_profiles = merged.get("sla_profiles_data", [])

    if not flows:
        return None, source, "no_flows"

    if not service_types:
        service_types = _default_service_types(flows)

    if not nodes:
        # 最小可运行拓扑: 1 CN + 1 AN
        nodes = [
            {
                "name": "CN_default",
                "id": 0,
                "type": "CN",
                "cpu_capacity": 1000.0,
                "memory_capacity": 1000.0,
                "mec_capacity": 200.0,
                "prb_capacity": 0.0,
                "slices_hosted": [],
            },
            {
                "name": "AN_default",
                "id": 1,
                "type": "AN",
                "cpu_capacity": 500.0,
                "memory_capacity": 500.0,
                "mec_capacity": 100.0,
                "prb_capacity": 135.0,
                "slices_hosted": [],
            },
        ]

    return {
        "flows_data": flows,
        "nodes_data": nodes,
        "service_types_data": service_types,
        "sla_profiles_data": sla_profiles,
    }, source, "ok"


def _simulate_ran_allocation_for_policy(
    policy_id: str,
    payload: Dict[str, Any],
    runner: Optional[Any] = None,
) -> Dict[str, Any]:
    """收到策略后触发一次RAN资源分配模拟。"""
    joint_inputs, source, prepare_status = _prepare_joint_inputs(payload)
    ts = time.time()

    if not joint_inputs:
        result = {
            "policy_id": policy_id,
            "timestamp": ts,
            "input_source": source,
            "ran_status": "skipped",
            "summary": f"RAN模拟跳过: {prepare_status}",
        }
        RAN_ALLOCATION_CACHE[policy_id] = result
        return result

    try:
        if runner is None:
            from tools.ran_scheduler.joint_simulator import JointSimulator
            from tools.ran_scheduler.config import JointSimConfig

            # 关键步骤: mock server 场景用 p1 模式提升兼容性，减少对SLA完整度的依赖
            cfg = JointSimConfig(cn_mode='p1', ran_num_steps=200, save_hist=False)
            sim = JointSimulator(cfg)

            def default_runner(inputs: Dict[str, Any]) -> Dict[str, Any]:
                return sim.run(
                    flows_data=inputs["flows_data"],
                    nodes_data=inputs["nodes_data"],
                    service_types_data=inputs["service_types_data"],
                    sla_profiles_data=inputs["sla_profiles_data"] or None,
                )

            runner = default_runner

        sim_result = runner(joint_inputs)
        overall = sim_result.get("ran_kpi", {}).get("overall", {}) if isinstance(sim_result, dict) else {}
        e2e = sim_result.get("e2e_feedback", {}) if isinstance(sim_result, dict) else {}
        status = sim_result.get("status", "error") if isinstance(sim_result, dict) else "error"

        result = {
            "policy_id": policy_id,
            "timestamp": ts,
            "input_source": source,
            "ran_status": "success" if status == "success" else status,
            "summary": e2e.get("summary", sim_result.get("message", "RAN simulation done") if isinstance(sim_result, dict) else "RAN simulation done"),
            "active_slices": int(overall.get("num_active_slices", 0)),
            "active_ues": int(overall.get("num_active_ues", 0)),
            "overall_sla_pass": bool(e2e.get("overall_sla_pass", False)),
        }
        RAN_ALLOCATION_CACHE[policy_id] = result
        return result
    except Exception as exc:
        result = {
            "policy_id": policy_id,
            "timestamp": ts,
            "input_source": source,
            "ran_status": "error",
            "summary": f"RAN模拟失败: {exc}",
        }
        RAN_ALLOCATION_CACHE[policy_id] = result
        return result

class MockPCFHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        """处理策略下发请求"""
        if self.path == '/pcf/policies':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                policy_id = data.get('policy_id', f"pol-{int(time.time())}")

                # 关键步骤: 在CN侧策略接收后，触发一次RAN侧资源分配模拟
                ran_result = _simulate_ran_allocation_for_policy(policy_id, data)
                print(
                    "[MockServer][RAN] "
                    f"policy_id={policy_id}, "
                    f"input_source={ran_result.get('input_source')}, "
                    f"ran_status={ran_result.get('ran_status')}, "
                    f"summary={ran_result.get('summary')}"
                )
                
                # 模拟处理延迟
                time.sleep(0.5)
                
                response = {
                    "code": 201,
                    "message": "Policy Created Successfully",
                    "data": {
                        "policy_id": policy_id,
                        "status": "ACTIVE"
                    }
                }
                
                self.send_response(201)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                print(f"[MockServer] 收到策略下发: ID={policy_id}")
                
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        """处理状态查询请求"""
        if self.path.startswith('/monitor/status/'):
            policy_id = self.path.split('/')[-1]
            ran_hint = RAN_ALLOCATION_CACHE.get(policy_id)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # 随机生成一些网络波动
            is_good = random.choice([True, True, True, False]) # 75% 概率正常
            
            feedback = {
                "policy_id": policy_id,
                "timestamp": time.time(),
                "monitoring_data": {
                    "throughput": "500 Mbps" if is_good else "20 Mbps",
                    "latency": "10ms" if is_good else "150ms",
                    "jitter": "2ms" if is_good else "25ms",
                    "packet_loss": "0.00%" if is_good else "2.5%"
                },
                "status": "COMPLIANT" if is_good else "NON_COMPLIANT"
            }
            
            self.wfile.write(json.dumps(feedback).encode('utf-8'))
            print(f"[MockServer] 返回监测数据 (Good={is_good}) for {policy_id}")
            if ran_hint:
                print(
                    "[MockServer][RAN-Cache] "
                    f"policy_id={policy_id}, "
                    f"ran_status={ran_hint.get('ran_status')}, "
                    f"summary={ran_hint.get('summary')}"
                )
        else:
            self.send_response(404)
            self.end_headers()

def run():
    server_address = (HOST, PORT)
    httpd = HTTPServer(server_address, MockPCFHandler)
    print(f"Mock PCF-SMF-UPF Server running on http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

if __name__ == '__main__':
    run()
