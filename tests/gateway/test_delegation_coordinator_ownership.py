"""Delegated execution must never acquire its coordinator's public route."""
import json
import time
import threading
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from agent.delegation_context import delegated_child_context
from gateway.config import GatewayConfig, Platform
from gateway.run_notifications import GatewayNotificationsMixin
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_state import AsyncSessionDB, SessionDB


@pytest.mark.asyncio
async def test_worker_notification_preserves_coordinator_handoff(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="ownership", user_id="owner")
    parent = store.get_or_create_session(source)
    db = store._db
    db.create_session("worker", source="telegram", parent_session_id=parent.session_id,
                      model_config={"_delegate_from": parent.session_id})
    runner = GatewayNotificationsMixin()
    runner.session_store = store
    runner.async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(db)
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 10
    accepted = []

    async def receive(event):
        current = store.get_or_create_session(event.source)
        target = await runner._resolve_async_delegation_session(
            current, event.metadata["gateway_session_id"])
        if target is not None:
            accepted.append((target.session_id, event.text))

    adapter = SimpleNamespace(handle_message=receive, supports_async_delivery=True)
    monkeypatch.setattr(runner, "_resolve_injection_adapter", lambda platform: adapter)
    event = {"type": "completion", "session_key": parent.session_key,
             "parent_session_id": "worker", "session_id": "process-worker"}
    await runner._inject_watch_notification("worker terminal finished", event)
    assert store.lookup_by_session_key(parent.session_key).session_id == parent.session_id
    assert db.get_session(parent.session_id)["ended_at"] is None
    assert not accepted
    from tools import async_delegation as ad
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    final = {"type": "async_delegation", "session_key": parent.session_key,
             "parent_session_id": parent.session_id, "delegation_id": "ownership-final"}
    ad._persist_dispatch({**final, "dispatched_at": time.time()})
    ad._persist_completion(final, {"summary": "true delegate result"})
    assert await runner._deliver_completion_notification("true delegate result", final) is True
    assert ad.get_durable_delegation("ownership-final")["delivery_state"] == "delivered"
    assert await runner._deliver_completion_notification("true delegate result", final) is None
    assert accepted == [(parent.session_id, "true delegate result")]
    assert store.get_or_create_session(source).session_id == parent.session_id
    # Explicit user resume remains an intentional route transition.
    assert store.switch_session(parent.session_key, "worker").session_id == "worker"
    store.close_all_db_handles()


@pytest.mark.parametrize("notify,patterns", [(True, None), (False, ["DONE"])])
def test_worker_terminal_notification_requires_polling(notify, patterns):
    from tools.terminal_tool_background import _apply_async_support
    tokens = set_session_vars(platform="telegram", session_id="parent", session_key="public", async_delivery=True)
    try:
        proc = SimpleNamespace(id="proc", watcher_platform="")
        result = {}
        with delegated_child_context("worker"):
            flags = _apply_async_support(proc, result, notify, patterns)
        assert flags == (False, None)
        assert result["notify_on_complete"] is False
        assert "poll" in result["notify_unsupported"]
        assert not proc.watcher_platform
        assert _apply_async_support(proc, {}, notify, patterns) == (notify, patterns)
    finally:
        clear_session_vars(tokens)


def test_nested_background_returns_result_to_worker(monkeypatch):
    from tools import delegate_tool_dispatch as dispatch
    from tools import delegate_tool as delegate
    tokens = set_session_vars(platform="telegram", session_id="parent", session_key="public", async_delivery=True)
    child = SimpleNamespace()
    task = {"goal": "nested work"}
    batch = dispatch._Batch([task], [(0, task, child)], SimpleNamespace(session_id="worker"),
                            {"model": "test"}, None, "leaf", 1, None, [], [], "worker", "", None, None, time.monotonic())
    monkeypatch.setattr(delegate, "_run_single_child", lambda **kw: {"task_index": 0, "summary": "nested result", "status": "completed"})
    monkeypatch.setattr(dispatch, "_finalize_child_results", lambda *args: None)
    def detached(**kwargs):
        pytest.fail("nested worker must not dispatch onto inherited public route")
    monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation_batch", detached)
    try:
        with delegated_child_context("worker"):
            result = json.loads(dispatch._dispatch_background(batch))
        assert result["results"][0]["summary"] == "nested result"
        assert "SYNCHRONOUSLY" in result["note"]
    finally:
        clear_session_vars(tokens)


@pytest.mark.parametrize("marker,explicit", [("_delegate_from", True), ("_branched_from", True),
                                             ("_delegate_from", False), ("_branched_from", False), (None, False)])
def test_lazy_child_routing_only_crosses_compression_edge(tmp_path, marker, explicit):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("parent", source="telegram", session_key="public", chat_id="chat")
        db.end_session("parent", "compression")
        config = {marker: "parent" if explicit else "ancestor"} if marker else {}
        db.create_session("child", source="telegram", parent_session_id="parent", model_config=config)
        row = db.get_session("child")
        assert row["session_key"] == (None if explicit else "public")
        assert row["chat_id"] == (None if explicit else "chat")
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["compression", "reset", "unrelated", "cas_reset"])
async def test_completion_respects_real_route_boundaries(tmp_path, monkeypatch, boundary):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    try:
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="boundary", user_id="owner")
        parent = store.get_or_create_session(source)
        parent_id = parent.session_id
        db = store._db
        runner = GatewayNotificationsMixin()
        runner._session_db = AsyncSessionDB(db)
        runner.async_session_store = AsyncSessionStore(store)
        current = parent
        if boundary in {"compression", "cas_reset"}:
            db.end_session(parent_id, "compression")
            db.create_session("tip", source="telegram", parent_session_id=parent_id)
            if boundary == "cas_reset":
                advance = store.advance_compression_session
                def reset_before_advance(key, expected_id, target_id):
                    store.reset_session(key)
                    return advance(key, expected_id, target_id)
                monkeypatch.setattr(store, "advance_compression_session", reset_before_advance)
        elif boundary == "reset":
            current = store.reset_session(parent.session_key)
        else:
            current = store.get_or_create_session(
                SessionSource(platform=Platform.TELEGRAM, chat_id="another", user_id="owner"))
        expected = store.lookup_by_session_key(current.session_key).session_id
        resolved = await runner._resolve_async_delegation_session(current, parent_id)
        if boundary == "compression":
            assert resolved.session_id == "tip"
            assert db.get_session(parent_id)["end_reason"] == "compression"
        else:
            assert resolved is None
            routed = store.lookup_by_session_key(current.session_key).session_id
            if boundary == "cas_reset":
                assert routed not in {parent_id, "tip"}
                assert db.get_session(routed)["ended_at"] is None
            else:
                assert routed == expected
    finally:
        store.close_all_db_handles()
