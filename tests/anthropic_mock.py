"""Minimal Anthropic SSE mock for integration testing M1.

Usage: python tests/anthropic_mock.py <port>
Cargo side: ANTHROPIC_API_KEY=test cargo run -p agent-cli -- print -p hi \
    --base-url http://127.0.0.1:<port>/v1/messages
"""
import http.server
import json
import sys
import time

EVENTS = [
    ("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "usage": {"input_tokens": 10,
        "cache_creation_input_tokens": 2, "cache_read_input_tokens": 0}
    }}),
    ("content_block_start", {"type": "content_block_start", "index": 0,
                             "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "text_delta", "text": "Hello"}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "text_delta", "text": " World"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta",
                       "delta": {"stop_reason": "end_turn"},
                       "usage": {"output_tokens": 5}}),
    ("message_stop", {"type": "message_stop"}),
]


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        sys.stderr.write("RECEIVED_BODY: " + body.decode()[:600] + "\n")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for name, payload in EVENTS:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.05)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
    http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()