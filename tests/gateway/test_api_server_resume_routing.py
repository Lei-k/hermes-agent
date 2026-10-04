"""Regression for #31: restart must not pair execution history with a new provider."""

from types import SimpleNamespace
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.session import SessionSource, SessionStore
from hermes_cli import config as cli_config, runtime_provider
from providers import ProviderProfile, register_provider

CODEX = "openai-codex"
DIRECT = "claude-subscription-directsdk-experimental"
ROUTES = {
    CODEX: ("gpt-synthetic", "https://chatgpt.com/backend-api/codex"),
    DIRECT: ("claude-synthetic", "https://directsdk.example/v1"),
}


@pytest.fixture
def routing_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    register_provider(ProviderProfile(name=DIRECT, base_url=ROUTES[DIRECT][1],
                                      env_vars=("SYNTHETIC_CORE31_KEY",)))
    missing = set()

    class Pool:
        def __init__(self, provider):
            self.provider = provider

        def has_credentials(self):
            if self.provider in missing:
                raise runtime_provider.AuthError("synthetic credentials removed")
            return True

        def select(self, **kwargs):
            return SimpleNamespace(access_token="synthetic-" + self.provider,
                                   source="manual", base_url=ROUTES[self.provider][1])

    # Only credential access and inference are fake; config loading, provider resolution,
    # persistence, API prelude, executor and GatewayRunner rehydration are production code.
    monkeypatch.setattr(runtime_provider, "load_pool", Pool)
    dispatches = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]
            self.session_id = kwargs["session_id"]
            self.db = kwargs["session_db"]

        def run_conversation(self, user_message, **kwargs):
            dispatches.append((self.model, self.provider, self.base_url))
            self.db.ensure_session(self.session_id, "api_server", model=self.model)
            self.db.append_message(self.session_id, "user", user_message)
            self.db.append_message(self.session_id, "assistant", "synthetic answer")
            return {"final_response": "synthetic answer", "completed": True, "messages": []}

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

    def configure(provider):
        (tmp_path / "config.yaml").write_text(
            "model:\n  provider: " + provider + "\n  default: " + ROUTES[provider][0] + "\n",
            encoding="utf-8")
        cli_config._LOAD_CONFIG_CACHE.clear()
        cli_config._RAW_CONFIG_CACHE.clear()

    def restart():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.session_store = store
        runner._session_model_overrides = {}
        runner.config = None
        monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "synthetic-api-key"}))
        adapter._session_db = store._db
        adapter.gateway_runner = runner
        return runner, adapter

    return configure, restart, missing, dispatches, tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("use_key", [True, False])
@pytest.mark.parametrize("original", [CODEX, DIRECT])
@pytest.mark.parametrize("selection", ["none", "pin", "partial", "missing"])
async def test_api_resume_restart_uses_a_complete_route(routing_env, original, selection, use_key):
    configure, restart, missing, dispatches, home = routing_env
    current = DIRECT if original == CODEX else CODEX
    configure(original)
    old_runner, old_adapter = restart()
    source = SessionSource(platform=Platform.API_SERVER, chat_id="synthetic-chat", chat_type="dm")
    entry = old_runner.session_store.get_or_create_session(source)
    session_id, key = entry.session_id, entry.session_key
    old_adapter._session_db.ensure_session(session_id, "api_server", model=ROUTES[original][0])
    old_adapter._session_db.append_message(session_id, "user", "before restart")
    old_adapter._session_db.append_message(session_id, "assistant", "old answer")
    if selection != "none":
        override = {"model": ROUTES[original][0]}
        if selection != "partial":
            override.update(provider=original, base_url=ROUTES[original][1], api_key="never-persist-me")
        old_runner.session_store.set_model_override(key, override)
    old_runner.session_store.close_all_db_handles()
    configure(current)
    if selection == "missing":
        missing.add(original)
    runner, adapter = restart()
    assert runner is not old_runner and runner.session_store is not old_runner.session_store
    restored = runner.session_store.lookup_by_session_key(key)
    assert restored.session_id == session_id
    assert "never-persist-me" not in (home / "sessions" / "sessions.json").read_text()
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    expected_provider = original if selection == "pin" else current
    expected = (ROUTES[expected_provider][0], expected_provider, ROUTES[expected_provider][1])
    headers = {"Authorization": "Bearer synthetic-api-key"}
    if use_key:
        headers["X-Hermes-Session-Key"] = key
    try:
        async with TestClient(TestServer(app)) as client:
            for _ in range(2):
                response = await client.post(f"/api/sessions/{session_id}/chat",
                    headers=headers,
                    json={"message": "resume synthetic work"})
                payload = await response.json()
                assert response.status == 200, payload
                assert dispatches[-1] == expected
        model, runtime = runner._resolve_session_agent_runtime(session_key=key)
        assert (model, runtime["provider"], runtime["base_url"]) == expected
        if selection == "missing":
            assert runner.session_store.get_model_override(key)["provider"] == original
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_key", [True, False])
@pytest.mark.parametrize("selection", ["request", "unconfirmed", "lock", "custom_pin", "transport_route", "profile_roundtrip"])
async def test_api_explicit_selection_survives_restart_without_model_name_guessing(
    routing_env, selection, use_key, monkeypatch,
):
    if selection == "profile_roundtrip":
        await _profile_roundtrip(routing_env, monkeypatch, use_key)
        return
    configure, restart, missing, dispatches, home = routing_env
    configure(DIRECT)
    runner, adapter = restart()
    source = SessionSource(platform=Platform.API_SERVER, chat_id="explicit-chat", chat_type="dm")
    entry = runner.session_store.get_or_create_session(source)
    session_id, key = entry.session_id, entry.session_key
    # A custom endpoint can deliberately serve a GPT-looking name on this provider.
    model, endpoint = ROUTES[CODEX][0], ROUTES[DIRECT][1]
    adapter._session_db.ensure_session(session_id, "api_server", model=model)
    if selection == "custom_pin":
        runner.session_store.set_model_override(key, {
            "model": model, "provider": DIRECT, "base_url": endpoint})
    elif selection in {"unconfirmed", "lock"}:
        adapter._session_db.update_session_runtime_lock(
            session_id, model=model, provider=DIRECT, confirmed=selection == "lock")
    runner.session_store.close_all_db_handles()
    configure(CODEX)
    runner, adapter = restart()
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    body = {"message": "continue explicit selection"}
    if selection == "request":
        body.update(model=model, provider=DIRECT)
    elif selection == "transport_route":
        missing.add(DIRECT)
        adapter._model_routes["synthetic-route"] = {
            "model": model, "provider": DIRECT, "base_url": endpoint, "api_key": "synthetic-route-key"}
        body["model"] = "synthetic-route"
    headers = {"Authorization": "Bearer synthetic-api-key"}
    if use_key:
        headers["X-Hermes-Session-Key"] = key
    try:
        async with TestClient(TestServer(app)) as client:
            response = await client.post(f"/api/sessions/{session_id}/chat",
                headers=headers, json=body)
            payload = await response.json()
            assert response.status == 200, payload
            assert dispatches[-1] == (model, DIRECT, endpoint)
    finally:
        runner.session_store.close_all_db_handles()


async def _profile_roundtrip(routing_env, monkeypatch, use_key):
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from hermes_cli.profiles import get_profile_dir

    configure, restart, missing, dispatches, home_a = routing_env
    monkeypatch.setattr(Path, "home", lambda: home_a)
    home_b = get_profile_dir("b")
    home_b.mkdir(parents=True)
    for home in (home_a, home_b):
        (home / ".env").write_text("API_SERVER_KEY=synthetic-api-key\n", encoding="utf-8")
    configure(DIRECT)
    (home_b / "config.yaml").write_text(
        f"model:\n  provider: {CODEX}\n  default: {ROUTES[CODEX][0]}\n", encoding="utf-8")
    cfg = GatewayConfig(multiplex_profiles=True)
    store = SessionStore(home_a / "sessions", cfg)
    shared_id = "same-id-in-independent-profile-dbs"
    keys = {}
    was_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for name, home, provider in [("default", home_a, CODEX), ("b", home_b, DIRECT)]:
            with gateway_run._profile_runtime_scope(home, prepared_secret_scope={}):
                source = SessionSource(platform=Platform.API_SERVER, chat_id="profile-chat", chat_type="dm", profile=name)
                entry = store.get_or_create_session(source)
                store._db.ensure_session(shared_id, "api_server", model=ROUTES[provider][0])
                store.switch_session(entry.session_key, shared_id)
                store.set_model_override(entry.session_key, {
                    "model": ROUTES[provider][0], "provider": provider, "base_url": ROUTES[provider][1]})
                keys[name] = entry.session_key
        store.close_all_db_handles()
        store = SessionStore(home_a / "sessions", cfg)
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.session_store, runner._session_model_overrides = store, {}
        runner.config = cfg
        monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
        for name, home, provider in [("default", home_a, CODEX), ("b", home_b, DIRECT), ("default", home_a, CODEX)]:
            with gateway_run._profile_runtime_scope(home, prepared_secret_scope={}):
                adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "synthetic-api-key"}))
                adapter.gateway_runner = runner
                adapter._session_db = store._db
                app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
                app.router.add_post("/p/{profile}/api/sessions/{session_id}/chat", adapter._handle_session_chat)
                headers = {"Authorization": "Bearer synthetic-api-key"}
                if use_key:
                    headers["X-Hermes-Session-Key"] = keys[name]
                async with TestClient(TestServer(app)) as client:
                    response = await client.post(f"/p/{name}/api/sessions/{shared_id}/chat", headers=headers,
                        json={"message": "profile-scoped resume"})
                    payload = await response.json()
                    assert response.status == 200, payload
                    assert dispatches[-1] == (ROUTES[provider][0], provider, ROUTES[provider][1])
    finally:
        store.close_all_db_handles()
        set_multiplex_active(was_multiplex)
