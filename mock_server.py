import time
import random
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# 配置
HOST = 'localhost'
PORT = 8000

class MockPCFHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        """处理策略下发请求"""
        if self.path == '/pcf/policies':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                policy_id = data.get('policy_id', f"pol-{int(time.time())}")
                
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
