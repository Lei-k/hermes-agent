"""Tests for gateway/wake.py — background wake delivery.

Two strategies:
* push-capable adapters keep the synthetic MessageEvent / handle_message path;
* the stateless API server (supports_async_delivery=False) self-POSTs
  /v1/chat/completions with the RAW session id in X-Hermes-Session-Id, so the
  wake turn resumes the REAL session instead of a parallel invisible one
  keyed by build_session_key().
"""

import asyncio

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.wake import deliver_wake, adapter_supports_push


class PushAdapter:
    """Default adapter shape — no supports_async_delivery attribute."""

    def __init__(self):
        self.handled = []

    async def handle_message(self, event):
        self.handled.append(event)


class ApiServerLikeAdapter:
    supports_async_delivery = False

    def __init__(self, host="0.0.0.0", port=0, key="test-key", model="hermes"):
        self._host = host
        self._port = port
        self._api_key = key
        self._model_name = model

    async def handle_message(self, event):  # pragma: no cover — must NOT be hit
        raise AssertionError("non-push adapter must not receive handle_message wakes")


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="group",
    )


def test_adapter_supports_push_default_true():
    assert adapter_supports_push(PushAdapter()) is True
    assert adapter_supports_push(ApiServerLikeAdapter()) is False


async def _serve(handler, path="/v1/chat/completions"):
    """Spin an in-process aiohttp server on an ephemeral loopback port."""
    from aiohttp import web

    app = web.Application()
    app.router.add_post(path, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def test_deliver_wake_non_push_self_posts_raw_session_id(monkeypatch):
    """The self-post carries the RAW session id header + bearer auth and a
    single user message with stream=false — the exact entry point real
    gateway turns use."""
    from aiohttp import web

    seen = {}

    async def handler(request):
        seen["session_id"] = request.headers.get("X-Hermes-Session-Id")
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = await request.json()
        return web.json_response({
            "choices": [{"message": {"content": "ok"}}],
            "hermes": {"completed": True, "failed": False, "partial": False},
        })

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(host="0.0.0.0", port=port, key="sekrit")
            await deliver_wake(adapter, text="task done — wake", session_id="raw-sid-42")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen["session_id"] == "raw-sid-42"
    assert seen["auth"] == "Bearer sekrit"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"] == [
        {"role": "user", "content": "task done — wake"}
    ]


def test_deliver_wake_retries_429_then_succeeds(monkeypatch):
    """HTTP 429 (max_concurrent_runs cap) is transient — retried with backoff."""
    from aiohttp import web

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return web.json_response({"error": "busy"}, status=429)
        return web.json_response({
            "choices": [],
            "hermes": {"completed": True, "failed": False, "partial": False},
        })

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port)
            await deliver_wake(adapter, text="x", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 2


def test_durable_self_post_retries_with_stable_authenticated_identity(monkeypatch):
    """Every transport attempt carries one bounded stable delivery identity."""
    from aiohttp import web

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01,))
    seen = []
    delivery_id = "async-delegation:deleg_transport_loss"

    async def handler(request):
        seen.append({"headers": dict(request.headers), "body": await request.json()})
        if len(seen) == 1:
            return web.json_response({"error": "lost response"}, status=429)
        return web.json_response({
            "choices": [{"message": {"content": "ok"}}],
            "hermes": {"completed": True, "failed": False, "partial": False},
        })

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port, key="sekrit")
            await deliver_wake(
                adapter,
                text="completion",
                session_id="raw-session",
                durable_delivery_ids=[delivery_id],
            )
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert len(seen) == 2
    assert {item["headers"]["Idempotency-Key"] for item in seen} == {delivery_id}
    assert {
        item["headers"]["X-Hermes-Durable-Delivery-Id"] for item in seen
    } == {delivery_id}
    assert {
        tuple(item["body"]["hermes_durable_delivery_ids"]) for item in seen
    } == {(delivery_id,)}


@pytest.mark.parametrize(
    "hermes_status",
    [
        {},
        {"completed": False, "failed": False, "partial": False},
        {"completed": True, "failed": True, "partial": False},
        {"completed": True, "failed": False, "partial": True},
    ],
)
def test_api_server_self_post_requires_explicit_clean_completion(
    monkeypatch, hermes_status
):
    """Missing, failed, or partial status cannot ACK a self-posted wake."""
    from aiohttp import web
    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", ())

    async def handler(_request):
        return web.json_response({"choices": [], "hermes": hermes_status})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port, key="sekrit")
            with pytest.raises(RuntimeError, match="incomplete turn"):
                await deliver_wake(adapter, text="completion", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_named_profile_wake_uses_bounded_profile_route_and_profile_token():
    from aiohttp import web

    seen = {}

    async def handler(request):
        seen["path"] = request.path
        seen["auth"] = request.headers.get("Authorization")
        return web.json_response({
            "choices": [],
            "hermes": {"completed": True, "failed": False, "partial": False},
        })

    async def run():
        runner, port = await _serve(handler, "/p/coder/v1/chat/completions")
        try:
            adapter = ApiServerLikeAdapter(port=port, key="listener-key")
            adapter.resolve_durable_wake_route = lambda profile: (
                "/p/coder/v1/chat/completions",
                "coder-profile-key",
            )
            await deliver_wake(
                adapter,
                text="named completion",
                session_id="named-session",
                origin_profile="coder",
            )
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen == {
        "path": "/p/coder/v1/chat/completions",
        "auth": "Bearer coder-profile-key",
    }


def test_invalid_or_unserved_profile_wake_fails_without_default_fallback():
    adapter = ApiServerLikeAdapter(key="listener-key")
    calls = []

    def reject(profile):
        calls.append(profile)
        raise ValueError("profile is not served")

    adapter.resolve_durable_wake_route = reject
    with pytest.raises(ValueError, match="not served"):
        asyncio.run(
            deliver_wake(
                adapter,
                text="completion",
                session_id="named-session",
                origin_profile="../default",
            )
        )
    assert calls == ["../default"]


