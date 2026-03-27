import json
import random
import time

import requests
from langchain_core.tools import tool

from tools.db_tool import get_ue_flow_catalog_by_supi
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Mock PCF base address used by the local integration environment.
PCF_BASE_URL = "http://localhost:8000"
PCF_DISPATCH_MAX_ATTEMPTS = 3
PCF_DISPATCH_RETRY_INTERVAL_SEC = 3


def _should_retry_dispatch(*, response=None, error=None) -> bool:
    if error is not None:
        return isinstance(
            error,
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ),
        )

    if response is None:
        return False

    if response.status_code in (401, 403, 408, 409, 429):
        return True
    if 500 <= response.status_code < 600:
        return True
    return False


def _build_pcf_dispatch_payload(policy_type: str, policy_details) -> dict:
    if not isinstance(policy_details, dict):
        raise ValueError("policy payload is not a JSON object")

    # Keep the original fields at the top level for servers that read policy_id,
    # pccRules, qosDecs, etc. directly from the request body, while still
    # preserving the wrapped structure used by the local agents.
    payload = dict(policy_details)
    payload.setdefault("policy_type", policy_type)
    payload["policy_details"] = policy_details
    return payload


def dispatch_policy_to_pcf(policy_type: str, policy_json: str) -> str:
    """
    Dispatch a policy payload to PCF through HTTP POST.

    Returns a plain-text result string consumed by PDA.
    """
    policy_details = json.loads(policy_json) if isinstance(policy_json, str) else policy_json
    payload = {"policy_type": policy_type, "policy": policy_details}
    # print(f"Dispatching to PCF with payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")

    if not str(PCF_BASE_URL or "").strip():
        return (
            f"策略下发失败: PCF address not configured after "
            f"{PCF_DISPATCH_MAX_ATTEMPTS} attempts"
        )

    # logger.info(
    #     f"Dispatching policy to PCF [{policy_type}]: "
    #     f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    # )

    last_error = None
    last_response = None
    for attempt in range(1, PCF_DISPATCH_MAX_ATTEMPTS + 1):
        response = requests.post(f"{PCF_BASE_URL}/pcf/policies", json=payload, timeout=5)
        last_response = response

        if response.ok:
            result = response.json()
            if attempt > 1:
                logger.info("PCF dispatch succeeded on retry #%s.", attempt)
            return f"策略下发成功. PCF 响应: {json.dumps(result, ensure_ascii=False)}"

        logger.warning(
            f"PCF dispatch rejected on attempt {attempt}/{PCF_DISPATCH_MAX_ATTEMPTS} "
            f"with status {response.status_code}: {response.text}"
        )
        if attempt < PCF_DISPATCH_MAX_ATTEMPTS and _should_retry_dispatch(response=response):
            time.sleep(PCF_DISPATCH_RETRY_INTERVAL_SEC)
            continue
        break
    if last_response is not None:
        return (
            f"策略下发失败: HTTP {last_response.status_code} after "
            f"{PCF_DISPATCH_MAX_ATTEMPTS} attempts. Response: {last_response.text}"
        )
    if last_error is not None:
        return (
            f"策略下发失败: {str(last_error)} after "
            f"{PCF_DISPATCH_MAX_ATTEMPTS} attempts"
        )
    return f"策略下发失败: unknown error after {PCF_DISPATCH_MAX_ATTEMPTS} attempts"


def get_network_feedback(policy_id: str) -> str:
    """
    Query feedback for a policy from the monitoring side.
    """
    logger.info(f"Querying execution feedback for policy {policy_id}")

    try:
        response = requests.get(f"{PCF_BASE_URL}/monitor/status/{policy_id}", timeout=5)
        if response.status_code == 200:
            remote_data = response.json()
            return f"Status: Success\nRaw: {json.dumps(remote_data, ensure_ascii=False)}"
    except requests.exceptions.RequestException:
        pass

    is_congested = random.random() > 0.9
    metrics = {
        "actual_throughput_dl": "450 Mbps" if not is_congested else "50 Mbps",
        "latency": "15ms" if not is_congested else "120ms",
        "packet_loss": "0.01%" if not is_congested else "5.0%",
    }

    if not is_congested:
        return f"Status: Success\nMetrics: {json.dumps(metrics, ensure_ascii=False)}"
    if random.random() > 0.5:
        return f"Status: Partial Success\nReason: High Latency\nMetrics: {json.dumps(metrics, ensure_ascii=False)}"
    return f"Status: Failed\nReason: Congestion\nMetrics: {json.dumps(metrics, ensure_ascii=False)}"


@tool
def get_ue_context(supi: str) -> str:
    """
    Query UE context details by SUPI.
    """
    logger.info(f"Querying UE Context: {supi}")

    try:
        from tools.db_tool import get_ue_context_by_supi

        db_ctx = get_ue_context_by_supi(supi)
        if db_ctx:
            return f"UE Context Retrieved From DB:\n{json.dumps(db_ctx, ensure_ascii=False, indent=2)}"
    except Exception as exc:
        logger.warning(f"Failed to read UEContext from DB, fallback to remote/mock: {exc}")

    try:
        response = requests.get(f"{PCF_BASE_URL}/pcf/ue_context/{supi}", timeout=5)
        if response.status_code == 200:
            return f"UE Context Found:\n{json.dumps(response.json(), ensure_ascii=False, indent=2)}"
    except Exception:
        pass

    return f"UE Context Not Found for SUPI: {supi}"


@tool
def get_ue_flow_catalog(supi: str) -> str:
    """
    Return the app/flow catalog of a UE from the latest scenario snapshot.
    """
    logger.info(f"Querying UE flow catalog: {supi}")
    catalog = get_ue_flow_catalog_by_supi(supi)
    result = json.dumps(catalog, ensure_ascii=False, indent=2)
    return f"UE Flow Catalog Retrieved:\n {result}"
