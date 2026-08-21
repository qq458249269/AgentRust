"""Minimal OpenAI chat.completions SSE mock for M1 verification.

Usage: python tests/openai_mock.py <port>
Cargo: python tests/openai_mock.py 18081 &
cargo run -p agent-cli -- print -p hi --provider "openai chat" \
    --base-url http://127.0.0.1:18081/v1/chat/completions --api-key test
"""
import http.server
import json
import sys
import time

CHUNKS = [
    {"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
    {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
    {"choices": [{"delta": {"content": " OpenAI"}, "finish_reason": None}]},
    {"choices": [{"delta": {}, "finish_reason": "stop"}],
     "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}},
]


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        sys.stderr.write("RECEIVED: " + body.decode()[:400] + "\n")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for c in CHUNKS:
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.04)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18081
    http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()