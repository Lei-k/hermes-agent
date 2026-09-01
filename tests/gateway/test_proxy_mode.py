"""Tests for gateway proxy mode — forwarding messages to a remote API server."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.platforms.base import resolve_proxy_url
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(proxy_url=None):
    """Create a minimal GatewayRunner for proxy tests."""
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = MagicMock()
    runner.config.streaming = StreamingConfig()
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    return runner


def _make_source(platform=Platform.MATRIX):
    return SessionSource(
        platform=platform,
        chat_id="!room:server.org",
        chat_name="Test Room",
        chat_type="group",
        user_id="@user:server.org",
        user_name="testuser",
        thread_id=None,
    )


class _FakeSSEResponse:
    """Simulates an aiohttp response with SSE streaming."""

    def __init__(self, status=200, sse_chunks=None, error_text=""):
        self.status = status
        self._sse_chunks = sse_chunks or []
        self._error_text = error_text
        self.content = self

    async def text(self):
        return self._error_text

    async def iter_any(self):
        for chunk in self._sse_chunks:
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeSession:
    """Simulates an aiohttp.ClientSession with captured request args."""

    def __init__(self, response):
        self._response = response
        self.captured_url = None
        self.captured_json = None
        self.captured_headers = None

    def post(self, url, json=None, headers=None, **kwargs):
        self.captured_url = url
        self.captured_json = json
        self.captured_headers = headers
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _patch_aiohttp(session):
    """Patch aiohttp.ClientSession to return our fake session."""
    return patch(
        "aiohttp.ClientSession",
        return_value=session,
    )


class TestGetProxyUrl:
    """Test _get_proxy_url() config resolution."""

    def test_returns_none_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
        runner = _make_runner()
        with patch("gateway.run._load_gateway_config", return_value={}):
            assert runner._get_proxy_url() is None


    def test_reads_from_config_yaml(self, monkeypatch):
        monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
        runner = _make_runner()
        cfg = {"gateway": {"proxy_url": "http://10.0.0.1:8642"}}
        with patch("gateway.run._load_gateway_config", return_value=cfg):
            assert runner._get_proxy_url() == "http://10.0.0.1:8642"


class TestResolveProxyUrl:

    def test_no_proxy_bypasses_matching_host(self, monkeypatch):
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                    "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
        monkeypatch.setenv("NO_PROXY", "api.telegram.org")

        assert resolve_proxy_url(target_hosts="api.telegram.org") is None

    def test_no_proxy_bypasses_cidr_target(self, monkeypatch):
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                    "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
        monkeypatch.setenv("NO_PROXY", "149.154.160.0/20")

        assert resolve_proxy_url(target_hosts=["149.154.167.220"]) is None


class TestRunAgentProxyDispatch:
    """Test that _run_agent() delegates to proxy when configured."""

    @pytest.mark.asyncio
    async def test_run_agent_delegates_to_proxy(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        runner = _make_runner()
        source = _make_source()

        expected_result = {
            "final_response": "Hello from remote!",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello from remote!"},
            ],
            "api_calls": 1,
            "tools": [],
        }

        runner._run_agent_via_proxy = AsyncMock(return_value=expected_result)

        result = await runner._run_agent(
            message="hi",
            context_prompt="",
            history=[],
            source=source,
            session_id="test-session-123",
            session_key="test-key",
            run_generation=7,
            durable_delivery_ids=["async-delegation:deleg_proxy_dispatch"],
        )

        assert result["final_response"] == "Hello from remote!"
        runner._run_agent_via_proxy.assert_called_once()
        assert runner._run_agent_via_proxy.call_args.kwargs["run_generation"] == 7
        assert runner._run_agent_via_proxy.call_args.kwargs[
            "durable_delivery_ids"
        ] == ["async-delegation:deleg_proxy_dispatch"]


class TestRunAgentViaProxy:
    """Test the actual proxy HTTP forwarding logic."""

    @pytest.mark.asyncio
    async def test_builds_correct_request(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.setenv("GATEWAY_PROXY_KEY", "test-key-123")
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(
            status=200,
            sse_chunks=[
                'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
                "data: [DONE]\n\n"
            ],
        )
        session = _FakeSession(resp)

        with patch("gateway.run._load_gateway_config", return_value={}):
            with _patch_aiohttp(session):
                with patch("aiohttp.ClientTimeout"):
                    result = await runner._run_agent_via_proxy(
                        message="How are you?",
                        context_prompt="You are helpful.",
                        history=[
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi there!"},
                        ],
                        source=source,
                        session_id="session-abc",
                        durable_delivery_ids=["async-delegation:deleg_proxy_wire"],
                    )

        # Verify request URL
        assert session.captured_url == "http://host:8642/v1/chat/completions"

        # Verify auth header
        assert session.captured_headers["Authorization"] == "Bearer test-key-123"

        # Verify session ID header
        assert session.captured_headers["X-Hermes-Session-Id"] == "session-abc"
        assert session.captured_headers["Idempotency-Key"] == (
            "async-delegation:deleg_proxy_wire"
        )
        assert session.captured_headers["X-Hermes-Durable-Delivery-Id"] == (
            "async-delegation:deleg_proxy_wire"
        )
        assert session.captured_json["hermes_durable_delivery_ids"] == [
            "async-delegation:deleg_proxy_wire"
        ]

        # Verify messages include system, history, and current message
        messages = session.captured_json["messages"]
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Hello"}
        assert messages[2] == {"role": "assistant", "content": "Hi there!"}
        assert messages[3] == {"role": "user", "content": "How are you?"}

        # Verify streaming is requested
        assert session.captured_json["stream"] is True

        # Verify response was assembled
        assert result["final_response"] == "Hello world"
        assert result["completed"] is True


    @pytest.mark.asyncio
    async def test_handles_connection_error(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://unreachable:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        runner = _make_runner()
        source = _make_source()

        class _ErrorSession:
            def post(self, *args, **kwargs):
                raise ConnectionError("Connection refused")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch("gateway.run._load_gateway_config", return_value={}):
            with patch("aiohttp.ClientSession", return_value=_ErrorSession()):
                with patch("aiohttp.ClientTimeout"):
                    result = await runner._run_agent_via_proxy(
                        message="hi",
                        context_prompt="",
                        history=[],
                        source=source,
                        session_id="test",
                    )

        assert "Proxy connection error" in result["final_response"]
        assert result["completed"] is False

    @pytest.mark.asyncio
    async def test_lost_response_retry_has_one_remote_effect_and_terminal_proof(
        self, monkeypatch
    ):
        """Retry reuses remote admission identity; only [DONE] is terminal proof."""
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.setenv("GATEWAY_PROXY_KEY", "test-key-123")
        runner = _make_runner()
        source = _make_source()
        delivery_id = "async-delegation:deleg_proxy_loss"
        admitted = set()
        model_effects = 0
        requests = []

        class LostResponse(_FakeSSEResponse):
            async def iter_any(self):
                yield b'data: {"choices":[{"delta":{"content":"incorporated"}}]}\n\n'
                raise ConnectionError("response lost after remote commit")

        responses = [
            LostResponse(status=200),
            _FakeSSEResponse(
                status=200,
                sse_chunks=[
                    b'data: {"choices":[{"delta":{"content":"incorporated"}}]}\n\n',
                    b"data: [DONE]\n\n",
                ],
            ),
        ]

        class RetrySession:
            def post(self, _url, json=None, headers=None, **_kwargs):
                nonlocal model_effects
                requests.append((json, headers))
                identity = tuple(json["hermes_durable_delivery_ids"])
                if identity not in admitted:
                    admitted.add(identity)
                    model_effects += 1
                return responses.pop(0)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        with patch("gateway.run._load_gateway_config", return_value={}):
            with patch("aiohttp.ClientSession", return_value=RetrySession()):
                with patch("aiohttp.ClientTimeout"):
                    first = await runner._run_agent_via_proxy(
                        message="completion",
                        context_prompt="",
                        history=[],
                        source=source,
                        session_id="session-loss",
                        durable_delivery_ids=[delivery_id],
                    )
                    second = await runner._run_agent_via_proxy(
                        message="completion",
                        context_prompt="",
                        history=[],
                        source=source,
                        session_id="session-loss",
                        durable_delivery_ids=[delivery_id],
                    )

        assert first["completed"] is False
        assert second["completed"] is True
        assert model_effects == 1
        assert len(requests) == 2
        assert {request[1]["Idempotency-Key"] for request in requests} == {
            delivery_id
        }

    @pytest.mark.asyncio
    async def test_durable_resume_does_not_send_later_turn_to_proxy(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.setenv("GATEWAY_PROXY_KEY", "test-key-123")
        runner = _make_runner()
        delivery_id = "async-delegation:deleg_proxy_boundary"
        durable_user = {
            "role": "user",
            "content": "durable completion",
            "message_id": delivery_id,
            "display_metadata": {
                "hermes_completion_delivery_ids": [delivery_id]
            },
        }
        later_turn = [
            {"role": "user", "content": "unrelated request"},
            {"role": "assistant", "content": "unrelated response"},
        ]
        # A late assistant message still linked to the durable delivery id
        # (e.g. a delayed retry of the terminal). It arrives after the later
        # user boundary and must not retroactively repair the ordering.
        late_linked_terminal = {
            "role": "assistant",
            "content": "durable done (late)",
            "display_metadata": {
                "hermes_completion_delivery_ids": [delivery_id]
            },
        }
        history = [durable_user, *later_turn, late_linked_terminal]
        resp = _FakeSSEResponse(
            status=200,
            sse_chunks=[
                b'data: {"choices":[{"delta":{"content":"durable done"}}]}\n\n',
                b"data: [DONE]\n\n",
            ],
        )
        session = _FakeSession(resp)

        with patch("gateway.run._load_gateway_config", return_value={}):
            with _patch_aiohttp(session):
                with patch("aiohttp.ClientTimeout"):
                    result = await runner._run_agent(
                        message="durable completion",
                        context_prompt="",
                        history=history,
                        source=_make_source(),
                        session_id="proxy-boundary",
                        durable_delivery_ids=[delivery_id],
                    )

        assert session.captured_url is None
        assert session.captured_json is None
        assert result["completed"] is False
        assert result["failed"] is True
        assert result["error"] == "durable_turn_conflict"
        assert result["messages"] == history


    @pytest.mark.asyncio
    async def test_no_system_message_when_context_empty(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_PROXY_URL", "http://host:8642")
        monkeypatch.delenv("GATEWAY_PROXY_KEY", raising=False)
        runner = _make_runner()
        source = _make_source()

        resp = _FakeSSEResponse(
            status=200,
            sse_chunks=[b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'],
        )
        session = _FakeSession(resp)

        with patch("gateway.run._load_gateway_config", return_value={}):
            with _patch_aiohttp(session):
                with patch("aiohttp.ClientTimeout"):
                    await runner._run_agent_via_proxy(
                        message="hello",
                        context_prompt="",
                        history=[],
                        source=source,
                        session_id="test",
                    )

        # No system message should appear when context_prompt is empty
        messages = session.captured_json["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"


class TestEnvVarRegistration:
    """Verify GATEWAY_PROXY_URL and GATEWAY_PROXY_KEY are registered."""

    def test_proxy_url_in_optional_env_vars(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert "GATEWAY_PROXY_URL" in OPTIONAL_ENV_VARS
        info = OPTIONAL_ENV_VARS["GATEWAY_PROXY_URL"]
        assert info["category"] == "messaging"
        assert info["password"] is False

