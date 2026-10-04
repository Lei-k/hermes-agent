"""Regression for #31: restart must not pair execution history with a new provider."""

from types import SimpleNamespace
from pathlib import Path
import sqlite3

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
    class Dispatches(list):
        runtimes = None

    dispatches = Dispatches()
    dispatches.runtimes = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.runtime = kwargs
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]
            self.session_id = kwargs["session_id"]
            self.db = kwargs["session_db"]

        def run_conversation(self, user_message, **kwargs):
            dispatches.append((self.model, self.provider, self.base_url))
            dispatches.runtimes.append(self.runtime)
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
    with sqlite3.connect(home / "state.db") as conn:
        assert "never-persist-me" not in str(conn.execute("SELECT entry_json FROM gateway_routing").fetchall())
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
            if selection == "transport_route":
                response = await client.post(f"/api/sessions/{session_id}/chat", headers=headers,
                    json={"message": "retain configured route on next turn"})
                assert response.status == 200, await response.text()
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


@pytest.mark.asyncio
@pytest.mark.parametrize("original", [CODEX, DIRECT])
@pytest.mark.parametrize("selection", [
    "create_model", "first_body", "legacy_lock", "legacy_confirmed", "legacy_alias", "one_turn",
    "fork_lock", "fork_route", "fork_pin", "row_route", "gateway_route",
])
async def test_native_selection_is_a_durable_complete_route(routing_env, original, selection):
    configure, restart, missing, dispatches, home = routing_env
    configure(original)
    runner, adapter = restart()
    model = "arbitrary-selected-model"
    endpoint = "https://selected-route.example/v1" if selection in {"row_route", "fork_route", "gateway_route"} else ROUTES[original][1]
    sid = "selection-source"

    def app_for(adapter):
        app = web.Application()
        app.router.add_post("/api/sessions", adapter._handle_create_session)
        app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
        app.router.add_post("/api/sessions/{session_id}/fork", adapter._handle_fork_session)
        return app

    auth = {"Authorization": "Bearer synthetic-api-key"}
    async with TestClient(TestServer(app_for(adapter))) as client:
        body = {"id": sid}
        if selection in {"create_model", "fork_lock", "one_turn"}:
            body["model"] = model
        response = await client.post("/api/sessions", headers=auth, json=body)
        assert response.status == 201, await response.text()
        if selection in {"legacy_lock", "legacy_confirmed", "legacy_alias"}:
            adapter._session_db.update_session_runtime_lock(
                sid, model="legacy-alias" if selection == "legacy_alias" else model,
                confirmed=selection != "legacy_lock",
                route_source="model_routes" if selection == "legacy_alias" else None)
        elif selection in {"row_route", "fork_route", "gateway_route"}:
            adapter._session_db.update_session_model(
                sid, model, original, base_url=endpoint, api_mode="chat_completions")
            if selection == "gateway_route":
                adapter._session_db.patch_session_model_config(sid, {"provider": None, "base_url": None})
        elif selection == "fork_pin":
            entry = runner.session_store.get_or_create_session(
                SessionSource(platform=Platform.API_SERVER, chat_id="fork-pin", chat_type="dm"))
            runner.session_store.switch_session(entry.session_key, sid)
            runner.session_store.set_model_override(entry.session_key, {
                "model": model, "provider": original, "base_url": endpoint, "api_key": "never-persist-fork-key"})
        if selection == "first_body":
            response = await client.post(f"/api/sessions/{sid}/chat", headers=auth,
                json={"message": "select on first turn", "model": model})
            assert response.status == 200, await response.text()
            assert dispatches[-1] == (model, original, ROUTES[original][1])
        if selection == "one_turn":
            other = DIRECT if original == CODEX else CODEX
            response = await client.post(f"/api/sessions/{sid}/chat", headers=auth,
                json={"message": "one turn switch", "model": ROUTES[other][0], "provider": other})
            assert response.status == 200, await response.text()
            assert dispatches[-1] == (ROUTES[other][0], other, ROUTES[other][1])
        if selection.startswith("fork_"):
            response = await client.post(f"/api/sessions/{sid}/fork", headers=auth, json={"id": "selection-fork"})
            assert response.status == 201, await response.text()
            sid = "selection-fork"
            if selection == "fork_pin":
                # A fork owns its copied selection even when the parent's gateway pin is reset.
                runner.session_store.set_model_override(entry.session_key, None)
    runner.session_store.close_all_db_handles()
    current = DIRECT if original == CODEX else CODEX
    configure(current)
    runner, adapter = restart()
    if selection == "legacy_alias":
        adapter._model_routes["legacy-alias"] = {"model": model, "provider": original}
    try:
        async with TestClient(TestServer(app_for(adapter))) as client:
            for _ in range(2):
                response = await client.post(f"/api/sessions/{sid}/chat", headers=auth,
                    json={"message": "resume selected route"})
                assert response.status == 200, await response.text()
                expected = (ROUTES[current][0], current, ROUTES[current][1]) if selection.startswith("legacy_") else (
                    model, original, endpoint)
                assert dispatches[-1] == expected
                if selection in {"row_route", "fork_route", "gateway_route"}:
                    assert dispatches.runtimes[-1]["api_mode"] == "chat_completions"
        stored = adapter._session_db.get_session(sid)["model_config"] or ""
        assert "synthetic-" + original not in stored
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["live_pin", "missing_route", "messaging_owner"])
async def test_api_route_credentials_belong_to_the_selected_provider(routing_env, selection, monkeypatch):
    configure, restart, missing, dispatches, home = routing_env
    configure(DIRECT)
    runner, adapter = restart()
    platform = Platform.TELEGRAM if selection == "messaging_owner" else Platform.API_SERVER
    entry = runner.session_store.get_or_create_session(
        SessionSource(platform=platform, chat_id="route-owner", chat_type="dm"))
    adapter._session_db.ensure_session(entry.session_id, "api_server")
    runner._session_model_overrides[entry.session_key] = {
        "model": ROUTES[CODEX][0], "provider": CODEX, "api_key": "synthetic-live-pin",
        "base_url": ROUTES[CODEX][1], "api_mode": "codex_responses"}
    auth = {"Authorization": "Bearer synthetic-api-key"}
    body = {"message": "check route credentials"}
    if selection == "live_pin":
        auth["X-Hermes-Session-Key"] = entry.session_key
        original_resolve = gateway_run._resolve_runtime_agent_kwargs

        def default_with_command():
            return {**original_resolve(), "command": "stale-default-command", "args": ["stale-default-arg"]}

        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", default_with_command)
    elif selection == "missing_route":
        missing.add(CODEX)
        # A route can use an arbitrary model name; auth must fail before inference.
        adapter._model_routes["unavailable-route"] = {
            "model": "arbitrary-selected-model", "provider": CODEX, "base_url": ROUTES[CODEX][1]}
        body["model"] = "unavailable-route"
        runner._session_model_overrides.clear()
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    try:
        async with TestClient(TestServer(app)) as client:
            response = await client.post(f"/api/sessions/{entry.session_id}/chat", headers=auth, json=body)
            payload = await response.json()
            assert response.status == 200, payload
            if selection == "missing_route":
                assert not dispatches
                assert "authentication failed" in payload["message"]["content"].lower()
            elif selection == "live_pin":
                assert dispatches[-1] == (ROUTES[CODEX][0], CODEX, ROUTES[CODEX][1])
                runtime = dispatches.runtimes[-1]
                assert runtime["credential_pool"].provider == CODEX
                assert not runtime.get("command") and not runtime.get("args")
            else:
                assert dispatches[-1] == (ROUTES[DIRECT][0], DIRECT, ROUTES[DIRECT][1])
    finally:
        runner.session_store.close_all_db_handles()
