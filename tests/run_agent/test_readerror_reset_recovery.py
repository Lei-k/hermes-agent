"""Reset recovery, proven against a real socket that really resets the connection.

Live evidence (2026-09-06): 64 of 67 failed provider attempts surfaced as a bare
``httpx.ReadError: [Errno 104] Connection reset by peer``. The Responses/Codex route
consumes the SSE body itself, so a mid-response reset escapes the OpenAI SDK
un-wrapped — the retry loop sees ``ReadError``, not ``APIConnectionError``.

``try_recover_primary_transport`` is the one place that retires the poisoned httpx
connection pool and rebuilds the client before giving up, and it is gated on a typed
allow-list. If that list disagrees with the canonical transport classifier, the exact
error the fleet actually hits skips the rebuild.

This test injects a REAL reset from a local listener (``SO_LINGER 0`` → RST): the
exception is produced by httpx, not constructed, and the recovery it drives goes on to
make a REAL successful request through the rebuilt client. No provider credentials.
"""

import json
import socket
import struct
import threading
import time

import httpx
import openai
import pytest
from unittest.mock import MagicMock, patch

from agent.error_classifier import classify_api_error
from run_agent import AIAgent

_COMPLETION_BODY = json.dumps(
    {
        "id": "chatcmpl-recovered",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "recovered"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
).encode()

# Generous enough that a loaded runner still has the client parked on the read when the
# RST lands; the whole harness pays it once per injected fault.
_ABORT_DELAY_SECONDS = 0.25


class ResettingEndpoint:
    """Local HTTP listener that aborts an ARMED response mid-stream with a TCP RST.

    Arming is explicit rather than "reset the first request" so the injected fault always
    lands on the request the test is measuring, whatever else happens to touch the port.
    """

    def __init__(self) -> None:
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.port = self._sock.getsockname()[1]
        self.connections = 0
        self.resets = 0
        self._armed = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def arm(self) -> None:
        """Reset the next model request; every request after it is served normally."""
        self._armed.set()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self.connections += 1
            try:
                request = conn.recv(65536)
                if b"/completions" in request and self._armed.is_set():
                    self._armed.clear()
                    self.resets += 1
                    self._abort(conn)
                else:
                    self._respond(conn)
            except OSError:
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _abort(self, conn: socket.socket) -> None:
        """Open the response, emit one frame, then RST the connection mid-body."""
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n"
            b'data: {"id":"c","object":"chat.completion.chunk","created":1,'
            b'"model":"test-model","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
        )
        # Let the client drain that frame and block on the next read first. Resetting while
        # the frame is still in flight races: the client can consume it and see the close as
        # a clean end-of-body, which is a *different* failure (silent truncation), not the
        # mid-read reset under test.
        time.sleep(_ABORT_DELAY_SECONDS)
        # SO_LINGER with a zero timeout makes close() send RST instead of FIN.
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))

    def _respond(self, conn: socket.socket) -> None:
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(_COMPLETION_BODY)).encode()
            + b"\r\nConnection: close\r\n\r\n"
            + _COMPLETION_BODY
        )

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass


@pytest.fixture
def endpoint():
    server = ResettingEndpoint()
    try:
        yield server
    finally:
        server.close()


def _make_agent(base_url, *, provider="custom"):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
    ):
        agent = AIAgent(
            api_key="test-key-12345678",
            base_url=base_url,
            provider=provider,
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    return agent


def _provoke_real_reset(endpoint, client) -> BaseException:
    """Drive a real request against the resetting endpoint and return what was raised."""
    endpoint.arm()
    with pytest.raises(BaseException) as excinfo:  # noqa: PT011 - the type IS the assertion
        stream = client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        for _chunk in stream:
            pass
    return excinfo.value


def test_local_fault_injection_reproduces_the_reported_error_shape(endpoint):
    """Sanity: the harness really produces the live bottom exception, unmocked."""
    client = openai.OpenAI(api_key="k", base_url=endpoint.base_url, max_retries=0)

    error = _provoke_real_reset(endpoint, client)

    assert isinstance(error, httpx.ReadError)
    assert "reset by peer" in str(error).lower()
    assert classify_api_error(error).retryable is True


def test_reset_drives_client_rebuild_and_a_successful_follow_up(endpoint):
    """The reported failure must reach the transport-recovery path and recover."""
    agent = _make_agent(endpoint.base_url)
    poisoned = openai.OpenAI(api_key="test-key-12345678", base_url=endpoint.base_url, max_retries=0)
    agent.client = poisoned
    retired: list = []
    agent._retire_shared_openai_client = lambda client, reason=None: retired.append(client)

    error = _provoke_real_reset(endpoint, poisoned)
    assert isinstance(error, httpx.ReadError)
    assert endpoint.resets == 1

    with patch("agent.agent_runtime_helpers.time.sleep"):
        recovered = agent._try_recover_primary_transport(error, retry_count=3, max_retries=3)

    assert recovered is True, "a real connection reset must get the one-shot client rebuild"
    assert retired == [poisoned], "the poisoned pool must be retired, not left checked out"
    assert agent.client is not poisoned

    # Credentials, model and route are unchanged by recovery.
    assert agent.model == "test-model"
    assert agent.provider == "custom"
    assert str(agent.client.base_url).rstrip("/") == endpoint.base_url.rstrip("/")
    assert agent.client.api_key == "test-key-12345678"

    # The rebuilt client is a working client, not just a new object.
    response = agent.client.chat.completions.create(
        model="test-model", messages=[{"role": "user", "content": "again"}]
    )
    assert response.choices[0].message.content == "recovered"
    assert endpoint.connections >= 2


def test_recovery_is_bounded_and_respects_client_ownership(endpoint):
    """Recovery never fires once fallback owns the client, and never for aggregators."""
    error = httpx.ReadError("[Errno 104] Connection reset by peer")

    on_fallback = _make_agent(endpoint.base_url)
    on_fallback.client = MagicMock()
    on_fallback._fallback_activated = True
    assert on_fallback._try_recover_primary_transport(error, retry_count=3, max_retries=3) is False

    aggregator = _make_agent("https://openrouter.ai/api/v1", provider="openrouter")
    aggregator.client = MagicMock()
    assert aggregator._try_recover_primary_transport(error, retry_count=3, max_retries=3) is False


def test_non_transport_errors_still_skip_the_rebuild(endpoint):
    """The fix must not turn the typed allow-list into "retry everything"."""
    agent = _make_agent(endpoint.base_url)
    agent.client = MagicMock()

    for error in (ValueError("bad request shape"), KeyError("model")):
        assert agent._try_recover_primary_transport(error, retry_count=3, max_retries=3) is False
