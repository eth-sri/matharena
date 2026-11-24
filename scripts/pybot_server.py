"""Serve PyBot behind an OpenAI-compatible /v1/chat/completions endpoint.

This lets MathArena's harness run PyBot as an ordinary `api: custom` model
(configs/models/pybot/pybot.yaml).

usage: python scripts/pybot_server.py PYBOT_DIR [PORT]

Every request is answered in a freshly forked process, so requests run in parallel and
cannot influence each other: the same as starting `python pybot.py -p MESSAGE` each time.
PyBot has no tokens, so the reported usage is zero.
"""
import json
import random
import socketserver
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

pybot_dir = sys.argv[1]
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765

sys.path.insert(0, pybot_dir)
from bot import runtime  # noqa: E402 - takes a few seconds: 640k lines of Python


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        # OpenAI's optional `seed` plays the role of `pybot.py --seed`; without it, wording is random.
        random.seed(body.get("seed"))
        memory = {}
        for message in body["messages"]:
            # PyBot keeps its own side of the conversation in `memory`, so replay the user's turns.
            if message["role"] == "user":
                reply = " ".join(str(runtime.turn(message["content"], memory)["reply"]).split())
        payload = json.dumps({
            "id": "chatcmpl-pybot", "object": "chat.completion", "created": int(time.time()), "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class Server(socketserver.ForkingMixIn, HTTPServer):
    max_children = 16


print("PyBot is listening on http://localhost:%d/v1" % port, flush=True)
Server(("127.0.0.1", port), Handler).serve_forever()
