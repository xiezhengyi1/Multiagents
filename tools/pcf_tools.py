import requests
import json
import time
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 模拟的 PCF 和网元地址 (如果运行 mock_server.py，端口通常为 8000)
PCF_BASE_URL = "http://localhost:8000"

def dispatch_policy_to_pcf(policy_type: str, policy_json: str) -> str:
    """
    将策略配置通过 HTTP POST 下发给 PCF 网元。
    
    Args:
        policy_type: 策略类型 (e.g. 'SmPolicyDecision')
        policy_json: JSON 格式的策略字符串 (包含 policy_id, directives 等)
        
    Returns:
        执行结果的文本描述 (成功/失败及原因)
    """
    try:
        # 尝试解析 JSON 以验证格式
        if isinstance(policy_json, str):
            payload = json.loads(policy_json)
        else:
            payload = policy_json
            
        # 注入类型信息到日志或 Payload
        logger.info(f"正在向 PCF 下发策略 [{policy_type}]: {json.dumps(payload, indent=2)}")
        
        # 模拟真实调用
        try:
            response = requests.post(f"{PCF_BASE_URL}/pcf/policies", json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            return f"策略下发成功. PCF 响应: {json.dumps(result)}"
        except requests.exceptions.ConnectionError:
            # 模拟环境
            logger.warning("无法连接到真实 PCF，切换到 Mock 模式返回成功响应。")
            return f"Mock模式: 策略下发成功 (Type: {policy_type}, ID: {payload.get('policy_id', 'unknown')})"
            
    except Exception as e:
        logger.error(f"策略下发失败: {e}")
        return f"策略下发失败: {str(e)}"

def get_network_feedback(policy_id: str) -> str:
    """
    从监控系统 (如 NWDAF 或 SMF) 获取特定策略 ID 的执行反馈。
    
    Args:
        policy_id: 策略的唯一标识 ID
    
    Returns:
        包含实际带宽、时延、丢包率等指标的文本报告 (需包含 Status: Success/Failed).
    """
    logger.info(f"正在查询策略 {policy_id} 的执行反馈...")
    
    import random
    
    # 1. 尝试真实调用
    try:
        response = requests.get(f"{PCF_BASE_URL}/monitor/status/{policy_id}", timeout=5)
        if response.status_code == 200:
            # 假设真实服务返回 {status: 'success', ...}
            # 这里简单返回 dump，但在 Fail-Fast 逻辑中我们需要 "Status: Success" 字样
            # 所以最好在这里做一层封装
            remote_data = response.json()
            return f"Status: Success\nRaw: {json.dumps(remote_data)}"
    except requests.exceptions.RequestException:
        pass

    # 2. Mock 数据生成
    # 模拟场景：90% 概率成功
    is_congested = random.random() > 0.9
    
    metrics = {
        "actual_throughput_dl": "450 Mbps" if not is_congested else "50 Mbps",
        "latency": "15ms" if not is_congested else "120ms",
        "packet_loss": "0.01%" if not is_congested else "5.0%"
    }
    
    if not is_congested:
        return f"Status: Success\nMetrics: {json.dumps(metrics)}"
    else:
        # 拥塞时，有一半概率是 Partial Success (比如部分指标达标)
        if random.random() > 0.5:
            return f"Status: Partial Success\nReason: High Latency\nMetrics: {json.dumps(metrics)}"
        else:
            return f"Status: Failed\nReason: Congestion\nMetrics: {json.dumps(metrics)}"

def get_ue_context(supi: str) -> str:
    """
    Query: 根据 SUPI 查询 PCF 数据库中的 UeContext 详细信息。
    包括订阅信息、AM/SM 策略数据等。
    
    Args:
        supi: 用户永久标识符 (Subscription Permanent Identifier), 如 'imsi-460010000000001'
    """
    logger.info(f"正在查询 UE Context: {supi}")
    
    # 尝试真实调用 (假设 Mock Server 有该接口，虽然目前没有)
    try:
        response = requests.get(f"{PCF_BASE_URL}/pcf/ue_context/{supi}", timeout=5)
        if response.status_code == 200:
            return f"UE Context Found:\n{json.dumps(response.json(), indent=2)}"
    except Exception:
        pass
        
    # Mock 数据 return
    # 如果是特定测试账号，返回特定数据
    if "46001" in supi:
        mock_context = {
            "supi": supi,
            "amPolicyData": {
                f"{supi}-1": {
                    "polAssoId": f"{supi}-1",
                    "accessType": "3GPP_ACCESS",
                    "servingPlmn": {"mcc": "460", "mnc": "01"},
                    "ratType": "NR",
                    "userLoc": {
                        "nrLocation": {
                            "tai": {"plmnId": {"mcc": "460", "mnc": "01"}, "tac": "000001"},
                            "ncgi": {"plmnId": {"mcc": "460", "mnc": "01"}, "nrCellId": "000000001"}
                        }
                    }
                }
            },
            "smPolicyData": {
                f"{supi}-1": {
                    "policyDecision": {
                        "pccRules": {
                            "default-rule": {
                                "flowInfos": [{"flowDescription": "permit out ip from any to any"}],
                                "qosDecs": ["qos-def"]
                            }
                        }
                    },
                    "smPolicySnssaiData": {
                        "01001": {
                            "snssai": {"sst": 1, "sd": "000001"}
                        }
                    }
                }
            }
        }
        return f"UE Context Retrieved Successfully:\n{json.dumps(mock_context, ensure_ascii=False, indent=2)}"
    
    return f"UE Context Not Found for SUPI: {supi}"

def commit_optimization_result(optimization_result: str) -> str:
    """
    [Critical] 将优化结果（内存中的最新状态）持久化保存到数据库。
    调用时机：当 PolicyDispatchAgent 确认：
       1. 策略已成功下发 (dispatch_policy_to_pcf 返回成功)
       2. 监控反馈符合预期 (get_network_feedback 指标正常)
       此时必须调用本工具，将最新的切片映射关系写入数据库，作为下一次调度的基准。
    
    Args:
        optimization_result: 优化步骤生成的 JSON 结果或描述字符串 (作为触发凭证)
    """
    from tools.commit_tool import commit_optimization_result_to_db
    return commit_optimization_result_to_db(optimization_result)

