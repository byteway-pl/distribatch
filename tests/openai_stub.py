"""Minimal OpenAI-compatible HTTP server for tests (no network, real HTTP).

Behaviour is driven by markers in the input text so tests stay declarative:
``RATE429:n`` / ``ERR500:n`` fail the first n calls for that input, ``BAD400`` always
returns 400. Chat echoes the last user message unless `chat_response` is set.
"""

from __future__ import annotations

import email.parser
import email.policy
import json
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_FAIL_N = re.compile(r"(RATE429|ERR500):(\d+)")


@dataclass
class StubState:
    chat_response: str | None = None
    reject_response_format: bool = False
    delay_s: float = 0.0
    calls: list[tuple[str, Any]] = field(default_factory=list)
    attempts: Counter[str] = field(default_factory=Counter)
    in_flight: int = 0
    max_in_flight: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def paths(self) -> list[str]:
        return [path for path, _ in self.calls]


class OpenAIStub:
    def __init__(self) -> None:
        self.state = StubState()
        state = self.state

        class Handler(_Handler):
            stub_state = state

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class _Handler(BaseHTTPRequestHandler):
    stub_state: StubState

    def log_message(self, format: str, *args: Any) -> None:  # silence test output
        pass

    def do_POST(self) -> None:
        st = self.stub_state
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with st.lock:
            st.in_flight += 1
            st.max_in_flight = max(st.max_in_flight, st.in_flight)
        try:
            if st.delay_s:
                time.sleep(st.delay_s)
            if self.path == "/v1/chat/completions":
                payload = json.loads(body)
                st.calls.append((self.path, payload))
                self._chat(payload)
            elif self.path == "/v1/embeddings":
                payload = json.loads(body)
                st.calls.append((self.path, payload))
                self._embeddings(payload)
            elif self.path in ("/v1/audio/transcriptions", "/v1/audio/translations"):
                fields = _multipart(self.headers["Content-Type"], body)
                st.calls.append((self.path, fields))
                self._audio(fields, translate=self.path.endswith("translations"))
            else:
                self._send(404, {"error": {"message": f"no route {self.path}"}})
        finally:
            with st.lock:
                st.in_flight -= 1

    # --- routes -----------------------------------------------------------------------

    def _chat(self, payload: dict[str, Any]) -> None:
        st = self.stub_state
        if st.reject_response_format and "response_format" in payload:
            self._send(400, _error("response_format of type json_schema is not supported"))
            return
        user = next(m["content"] for m in reversed(payload["messages"]) if m["role"] == "user")
        if "BAD400" in user:
            self._send(400, _error("bad request"))
            return
        if self._injected_failure(user):
            return
        text = st.chat_response if st.chat_response is not None else user
        words = sum(len(m["content"].split()) for m in payload["messages"])
        self._send(
            200,
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": words,
                    "completion_tokens": len(text.split()),
                    "total_tokens": words + len(text.split()),
                },
            },
        )

    def _embeddings(self, payload: dict[str, Any]) -> None:
        inputs = payload["input"] if isinstance(payload["input"], list) else [payload["input"]]
        if any("BAD400" in t for t in inputs):
            self._send(400, _error("input contains an invalid token"))
            return
        if self._injected_failure(inputs[0]):
            return
        dim = payload.get("dimensions") or 4
        data = [
            {"object": "embedding", "index": i, "embedding": [float(len(t))] + [1.0] * (dim - 1)}
            for i, t in enumerate(inputs)
        ]
        tokens = sum(len(t.split()) for t in inputs)
        self._send(
            200,
            {
                "object": "list",
                "data": data,
                "model": payload["model"],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            },
        )

    def _audio(self, fields: dict[str, Any], *, translate: bool) -> None:
        content = fields["file"][1].decode("utf-8", errors="replace")
        if self._injected_failure(content):
            return
        if "BAD400" in content:
            self._send(400, _error("could not decode audio"))
            return
        text = ("EN: " if translate else "") + content
        if fields.get("response_format") != "verbose_json":
            self._send(200, {"text": text})
            return
        self._send(
            200,
            {
                "task": "translate" if translate else "transcribe",
                "language": fields.get("language") or "english",
                "duration": 2.5,
                "text": text,
                "segments": [
                    {
                        "id": 0,
                        "seek": 0,
                        "start": 0.0,
                        "end": 2.5,
                        "text": text,
                        "tokens": [],
                        "temperature": 0.0,
                        "avg_logprob": 0.0,
                        "compression_ratio": 1.0,
                        "no_speech_prob": 0.0,
                    }
                ],
            },
        )

    # --- helpers ----------------------------------------------------------------------

    def _injected_failure(self, text: str) -> bool:
        st = self.stub_state
        m = _FAIL_N.search(text)
        if m is None:
            return False
        with st.lock:
            st.attempts[text] += 1
            attempt = st.attempts[text]
        if attempt > int(m[2]):
            return False
        if m[1] == "RATE429":
            self._send(429, _error("rate limited"), headers={"Retry-After": "0"})
        else:
            self._send(500, _error("internal error"))
        return True

    def _send(self, status: int, payload: Any, headers: dict[str, str] | None = None) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(raw)


def _error(message: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": "invalid_request_error"}}


def _multipart(content_type: str, body: bytes) -> dict[str, Any]:
    """Form fields; the file field is (filename, bytes)."""
    raw = f"Content-Type: {content_type}\r\n\r\n".encode() + body
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(raw)
    fields: dict[str, Any] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        if part.get_filename():
            fields[str(name)] = (part.get_filename(), payload)
        else:
            fields[str(name)] = payload.decode() if isinstance(payload, bytes) else payload
    return fields
