import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlsplit


HOST = "localhost"
PORT = 18080
GENERIC_POLICY_PATH = "/pcf/policies"
AM_POLICY_PATH = "/npcf-am-policy-control/v1/policies"
AM_POLICY_TYPE = "PcfAmPolicyControlPolicyAssociation"

POLICY_EXECUTION_CACHE: Dict[str, Dict[str, Any]] = {}
FAIL_ONCE_STATE = {
    "am": 1 if str(os.getenv("MOCK_FAIL_ONCE_AM", "")).strip() == "1" else 0,
}
FAIL_ONCE_STATUS = int(str(os.getenv("MOCK_FAIL_ONCE_AM_STATUS", "422")).strip() or "422")


def _coerce_non_empty(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _extract_flow_id(policy_details: Dict[str, Any]) -> str:
    direct_flow_id = str(policy_details.get("flow_id") or "").strip()
    if direct_flow_id:
        return direct_flow_id

    pcc_rules = policy_details.get("pccRules")
    if isinstance(pcc_rules, dict) and pcc_rules:
        first_key = next(iter(pcc_rules.keys()))
        first_rule = pcc_rules[first_key]
        if isinstance(first_rule, dict):
            flow_id = str(first_rule.get("flow_id") or "").strip()
            if flow_id:
                return flow_id
            pcc_rule_id = str(first_rule.get("pccRuleId") or first_key).strip()
            if pcc_rule_id.startswith("pcc-") and len(pcc_rule_id) > 4:
                return pcc_rule_id[4:]

    qos_decs = policy_details.get("qosDecs")
    if isinstance(qos_decs, dict) and qos_decs:
        first_key = str(next(iter(qos_decs.keys()))).strip()
        if first_key.startswith("qos-") and len(first_key) > 4:
            return first_key[4:]

    return ""


def _validate_dispatch_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")

    request_id = _coerce_non_empty(payload.get("request_id"), "request_id")
    session_id = str(payload.get("session_id") or "").strip()
    snapshot_id = str(payload.get("snapshot_id") or "").strip()
    policy_id = _coerce_non_empty(payload.get("policy_id"), "policy_id")
    policy_type = _coerce_non_empty(payload.get("policy_type"), "policy_type")
    policy_details = payload.get("policy_details")

    if not isinstance(policy_details, dict):
        raise ValueError("policy_details must be an object")

    nested_policy_id = str(policy_details.get("policy_id") or "").strip()
    if nested_policy_id and nested_policy_id != policy_id:
        raise ValueError("top-level policy_id does not match policy_details.policy_id")

    validated = {
        "request_id": request_id,
        "session_id": session_id,
        "snapshot_id": snapshot_id,
        "policy_id": policy_id,
        "policy_type": policy_type,
        "policy_details": policy_details,
        "flow_id": _extract_flow_id(policy_details),
    }
    if policy_type == AM_POLICY_TYPE:
        request = policy_details.get("request")
        if not isinstance(request, dict):
            raise ValueError("AM policy payload requires policy_details.request")
        if not str(request.get("supi") or "").strip():
            raise ValueError("AM policy payload requires request.supi")
    return validated


def _build_ack(payload: Dict[str, Any]) -> Dict[str, Any]:
    applied_at = time.time()
    event_type = "am_policy_applied" if payload["policy_type"] == AM_POLICY_TYPE else "policy_applied"
    return {
        "request_id": payload["request_id"],
        "policy_id": payload["policy_id"],
        "expected": 1,
        "received": 1,
        "completed": True,
        "results": [
            {
                "eventType": event_type,
                "policy_id": payload["policy_id"],
                "policy_type": payload["policy_type"],
                "session_id": payload["session_id"],
                "snapshot_id": payload["snapshot_id"],
                "flow_id": payload["flow_id"],
                "applied_at": applied_at,
                "status": "applied",
            }
        ],
    }


def _build_monitor_record(payload: Dict[str, Any], ack: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "success",
        "policy_id": payload["policy_id"],
        "policy_type": payload["policy_type"],
        "session_id": payload["session_id"],
        "snapshot_id": payload["snapshot_id"],
        "flow_id": payload["flow_id"],
        "timestamp": time.time(),
        "monitoring_data": {
            "ack_completed": ack["completed"],
            "applied_results": len(ack["results"]),
        },
        "compliance_status": "COMPLIANT",
    }


class MockPCFHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Optional[Dict[str, Any]]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        if not raw:
            raise ValueError("request body is empty")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        raw_path = self.path or "/"
        parsed = urlsplit(raw_path)
        normalized_path = (parsed.path or raw_path.split("?", 1)[0] or "/").rstrip("/") or "/"
        if normalized_path not in {GENERIC_POLICY_PATH, AM_POLICY_PATH}:
            self._send_json(404, {"status": "failed", "error": "not found", "raw_path": raw_path, "normalized_path": normalized_path})
            return

        try:
            raw_payload = self._read_json_body()
            payload = _validate_dispatch_payload(raw_payload)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"status": "failed", "error": str(exc)})
            return

        if payload["policy_type"] == AM_POLICY_TYPE and FAIL_ONCE_STATE["am"] > 0:
            FAIL_ONCE_STATE["am"] -= 1
            self._send_json(
                FAIL_ONCE_STATUS,
                {
                    "status": "failed",
                    "request_id": payload["request_id"],
                    "policy_id": payload["policy_id"],
                    "policy_type": payload["policy_type"],
                    "error": "simulated AM policy rejection for retry-path evaluation",
                },
            )
            return

        ack = _build_ack(payload)
        monitor_record = _build_monitor_record(payload, ack)
        POLICY_EXECUTION_CACHE[payload["policy_id"]] = {
            "payload": payload,
            "ack": ack,
            "monitor": monitor_record,
        }

        self._send_json(
            201,
            {
                "status": "success",
                "path": normalized_path,
                "request_id": payload["request_id"],
                "session_id": payload["session_id"],
                "snapshot_id": payload["snapshot_id"],
                "policy_id": payload["policy_id"],
                "policy_type": payload["policy_type"],
                "ack": ack,
                "message": "Policy accepted.",
            },
        )

    def do_GET(self) -> None:
        if not self.path.startswith("/monitor/status/"):
            self._send_json(404, {"status": "failed", "error": "not found"})
            return

        policy_id = self.path.rsplit("/", 1)[-1].strip()
        if not policy_id:
            self._send_json(400, {"status": "failed", "error": "policy_id is required"})
            return

        cached = POLICY_EXECUTION_CACHE.get(policy_id)
        if not cached:
            self._send_json(
                404,
                {
                    "status": "failed",
                    "policy_id": policy_id,
                    "error": "policy_id not found",
                },
            )
            return

        self._send_json(200, cached["monitor"])


def run() -> None:
    server_address = (HOST, PORT)
    httpd = HTTPServer(server_address, MockPCFHandler)
    print(f"Mock PCF server running on http://{HOST}:{PORT}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run()
