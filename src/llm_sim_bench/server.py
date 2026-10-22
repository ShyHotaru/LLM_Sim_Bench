"""Minimal OpenAI-compatible HTTP server backed by MockEngine (stdlib only).

Point any OpenAI client or chat UI at http://host:port/v1 to feel the
simulated speed before buying hardware. Replies stream a text preset: for a
chat preset, the reply to the k-th user message is the k-th scripted answer
(wrapping around), so a conversation in a chat UI follows the script.
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import texts
from .mock import MockEngine, StreamStats


def reply_for(script: texts.Script, messages: list) -> list[str]:
    """Assistant pieces for a request: the scripted answer to its latest user message."""
    users = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "user")
    return script.turns[(max(users, 1) - 1) % len(script.turns)][1]


def make_handler(engine: MockEngine, model_name: str, default_max_tokens: int,
                 script: texts.Script):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": model_name, "object": "model", "owned_by": "llm-sim-bench"}]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._json(404, {"error": {"message": "not found"}})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": {"message": "invalid JSON"}})

            messages = body.get("messages", [])
            prompt = "".join(_text(m.get("content")) for m in messages if isinstance(m, dict))
            prompt_tokens = max(1, texts.count_tokens(prompt, texts.prompt_ratio(
                prompt, script.family, script.ratio_id)))
            max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens")
                             or default_max_tokens)
            reply = reply_for(script, messages)
            finish = "stop" if len(reply) <= max_tokens else "length"
            stats = StreamStats()
            tokens = engine.stream(prompt_tokens, reply[:max_tokens], stats)
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    for i, tok in enumerate(tokens):
                        # Below one character per token some pieces are empty; like a
                        # real server holding back a partial character, send nothing.
                        if i == 0:
                            self._sse(cid, created, {"role": "assistant", "content": tok}, None)
                        elif tok:
                            self._sse(cid, created, {"content": tok}, None)
                    self._sse(cid, created, {}, finish)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.log_message("stream %s", _summary(stats))
                return

            text = "".join(tokens)
            self.log_message("complete %s", _summary(stats))
            self._json(200, {
                "id": cid, "object": "chat.completion", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "finish_reason": finish,
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": stats.prompt_tokens,
                          "completion_tokens": stats.completion_tokens,
                          "total_tokens": stats.prompt_tokens + stats.completion_tokens},
            })

        def _sse(self, cid, created, delta, finish):
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                     "model": model_name,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        def _json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multi-part content
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _summary(s: StreamStats) -> str:
    return (f"prompt={s.prompt_tokens} gen={s.completion_tokens} "
            f"ttft={s.ttft_s:.2f}s decode={s.decode_tps:.1f} tok/s")


def serve(engine: MockEngine, model_name: str, host: str, port: int,
          default_max_tokens: int = 256, script: texts.Script | None = None) -> None:
    script = script or texts.script("")
    httpd = ThreadingHTTPServer((host, port), make_handler(engine, model_name,
                                                           default_max_tokens, script))
    print(f"Mock OpenAI API at http://{host}:{port}/v1  (model: {model_name})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
