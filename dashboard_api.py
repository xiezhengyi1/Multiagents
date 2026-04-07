from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from tools.dashboard_data import (
    create_use_input_record,
    get_dashboard_overview,
    get_latest_network_snapshot,
    get_node_details,
    list_agent_statuses,
    list_queue_items,
    list_use_inputs,
)


HOST = "127.0.0.1"
PORT = 8765


def _json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_length)
    if not raw:
        return {}
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


class DashboardApiHandler(BaseHTTPRequestHandler):
    server_version = "MultiAgentsDashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status_code: int, message: str) -> None:
        self._send_json(status_code, {"error": message})

    def do_OPTIONS(self) -> None:
        self._send_json(204, {})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/api/dashboard/overview":
                self._send_json(200, get_dashboard_overview())
                return

            if path == "/api/dashboard/agents":
                self._send_json(200, {"items": list_agent_statuses()})
                return

            if path == "/api/dashboard/queues":
                limit = int(query.get("limit", ["100"])[0])
                self._send_json(200, {"items": list_queue_items(limit=limit)})
                return

            if path == "/api/dashboard/use-inputs":
                limit = int(query.get("limit", ["20"])[0])
                self._send_json(200, {"items": list_use_inputs(limit=limit)})
                return

            if path == "/api/dashboard/network":
                snapshot = get_latest_network_snapshot()
                if snapshot is None:
                    self._send_error(404, "No network graph snapshot found")
                    return
                self._send_json(200, snapshot)
                return

            if path.startswith("/api/dashboard/nodes/"):
                node_key = path.removeprefix("/api/dashboard/nodes/")
                details = get_node_details(node_key)
                if details is None:
                    self._send_error(404, f"Node not found: {node_key}")
                    return
                self._send_json(200, details)
                return

            self._send_error(404, f"Unknown endpoint: {path}")
        except ValueError as exc:
            self._send_error(400, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._send_error(500, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/dashboard/use-inputs":
            self._send_error(404, f"Unknown endpoint: {parsed.path}")
            return

        try:
            payload = _json_body(self)
            text = str(payload.get("text") or "").strip()
            session_id = str(payload.get("session_id") or "").strip()
            record = create_use_input_record(text=text, session_id=session_id)
            self._send_json(201, record)
        except ValueError as exc:
            self._send_error(400, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._send_error(500, str(exc))


def run(host: str = HOST, port: int = PORT) -> None:
    server = ThreadingHTTPServer((host, port), DashboardApiHandler)
    print(f"Dashboard API listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
