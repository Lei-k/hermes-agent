"""Regression coverage for #64484 — durable-restored delegation completions
must never be adopted by a session that cannot positively prove ownership.

Fixture timestamps are recent (now-based): restore_undelivered_completions
terminally drops pending completions older than _MAX_COMPLETION_REPLAY_AGE_S,
so epoch-era toy timestamps would exercise the staleness cap instead of the
restored-flag contract under test here.

Layers under test:
1. ``restore_undelivered_completions`` stamps every restored event with
   ``restored=True`` (in-memory only).
2. ``ProcessRegistry.drain_notifications`` with NO filter (legacy
   consume-everything CLI path) re-queues restored events instead of
   consuming them.
3. Same-process (non-restored) keyless events keep the legacy behavior.
4. An owner with a matching session_key still receives its restored event.
"""

import json
import queue
import time

from tools.process_registry import ProcessRegistry


def _make_registry():
    reg = ProcessRegistry.__new__(ProcessRegistry)
    import threading

    reg._running = {}
    reg._finished = {}
    reg._lock = threading.Lock()
    reg.completion_queue = queue.Queue()
    reg._completion_consumed = set()
    reg._poll_observed = set()
    return reg


def _delegation_event(session_key="", restored=False, delegation_id="d1"):
    evt = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": session_key,
        "origin_ui_session_id": "",
        "goal": "secret goal",
        "status": "success",
        "summary": "SECRET RESULT",
        "api_calls": 3,
        "duration_seconds": 1.5,
        "dispatched_at": time.time() - 2.0,
        "completed_at": time.time() - 1.0,
    }
    if restored:
        evt["restored"] = True
    return evt


def test_restore_stamps_restored_flag(tmp_path, monkeypatch):
    """Every durable completion re-enqueued at startup carries restored=True."""
    import tools.async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "async_delegations.db")
    record = {
        "delegation_id": "d-old",
        "goal": "old goal",
        "context": None,
        "toolsets": None,
        "role": "leaf",
        "model": "m",
        "session_key": "OLD_SESSION_A",
        "origin_ui_session_id": "",
        "parent_session_id": "OLD_SESSION_A",
        "status": "running",
        "dispatched_at": time.time() - 2.0,
        "completed_at": None,
        "interrupt_fn": None,
    }
    ad._persist_dispatch(record)
    evt = _delegation_event(session_key="OLD_SESSION_A", delegation_id="d-old")
    ad._persist_completion(evt, {"summary": "SECRET RESULT"})

    q = queue.Queue()
    restored = ad.restore_undelivered_completions(q)
    assert restored == 1
    got = q.get_nowait()
    assert got["restored"] is True
    assert got["session_key"] == "OLD_SESSION_A"

    # The stamp is in-memory only — the durable payload is unchanged.
    with ad._connect() as conn:
        row = conn.execute(
            "SELECT event_json FROM async_delegations WHERE delegation_id='d-old'"
        ).fetchone()
    assert "restored" not in json.loads(row[0])


def test_restore_releases_claim_owned_by_crashed_process(tmp_path, monkeypatch):
    """A gateway crash after claim must not strand the replay for five minutes."""
    import gateway.status as gateway_status
    import tools.async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "async_delegations.db")
    record = {
        "delegation_id": "d-crashed-claim",
        "goal": "old goal",
        "context": None,
        "toolsets": None,
        "role": "leaf",
        "model": "m",
        "session_key": "OLD_SESSION_A",
        "origin_ui_session_id": "",
        "parent_session_id": "OLD_SESSION_A",
        "status": "running",
        "dispatched_at": time.time() - 2.0,
        "completed_at": None,
        "interrupt_fn": None,
    }
    ad._persist_dispatch(record)
    evt = _delegation_event(
        session_key="OLD_SESSION_A",
        delegation_id="d-crashed-claim",
    )
    ad._persist_completion(evt, {"summary": "SECRET RESULT"})
    assert ad.claim_completion_delivery("d-crashed-claim", "dead-gateway-claim")

    monkeypatch.setattr(gateway_status, "_pid_exists", lambda _pid: False)
    q = queue.Queue()
    assert ad.restore_undelivered_completions(q) == 1
    restored = q.get_nowait()
    assert restored["delegation_id"] == "d-crashed-claim"
    assert restored["restored"] is True

    # Immediate reclaim is the crash-replay proof. Before the fix the old
    # claim blocks this for the hard-coded 300-second stale timeout.
    assert ad.claim_completion_delivery("d-crashed-claim", "restart-claim")
    assert ad.release_completion_delivery("d-crashed-claim", "restart-claim")


def test_repeated_orphan_restores_do_not_spend_delivery_attempts(
    tmp_path, monkeypatch,
):
    """A claim owner crash is not a delivery failure and never exhausts replay."""
    import gateway.status as gateway_status
    import tools.async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "async_delegations.db")
    record = {
        "delegation_id": "d-repeated-orphans",
        "goal": "old goal",
        "context": None,
        "toolsets": None,
        "role": "leaf",
        "model": "m",
        "session_key": "OLD_SESSION_A",
        "origin_ui_session_id": "",
        "parent_session_id": "OLD_SESSION_A",
        "status": "running",
        "dispatched_at": time.time() - 2.0,
        "completed_at": None,
        "interrupt_fn": None,
    }
    ad._persist_dispatch(record)
    evt = _delegation_event(
        session_key="OLD_SESSION_A",
        delegation_id="d-repeated-orphans",
    )
    ad._persist_completion(evt, {"summary": "SECRET RESULT"})
    monkeypatch.setattr(gateway_status, "_pid_exists", lambda _pid: False)

    for crash in range(ad._MAX_DELIVERY_ATTEMPTS + 2):
        assert ad.claim_completion_delivery(
            "d-repeated-orphans", f"dead-gateway-{crash}"
        )
        restored = queue.Queue()
        assert ad.restore_undelivered_completions(restored) == 1
        assert restored.get_nowait()["delegation_id"] == "d-repeated-orphans"

    durable = ad.get_durable_delegation("d-repeated-orphans")
    assert durable is not None
    assert durable["delivery_state"] == "pending"
    assert durable["delivery_attempts"] == 0


def test_live_process_claim_is_not_stolen_only_because_timestamp_is_old(
    tmp_path, monkeypatch,
):
    """Long busy receipts keep ownership until release; age alone is unsafe."""
    import tools.async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "async_delegations.db")
    record = {
        "delegation_id": "d-live-old-claim",
        "goal": "old goal",
        "context": None,
        "toolsets": None,
        "role": "leaf",
        "model": "m",
        "session_key": "OLD_SESSION_A",
        "origin_ui_session_id": "",
        "parent_session_id": "OLD_SESSION_A",
        "status": "running",
        "dispatched_at": time.time() - 2.0,
        "completed_at": None,
        "interrupt_fn": None,
    }
    ad._persist_dispatch(record)
    evt = _delegation_event(
        session_key="OLD_SESSION_A",
        delegation_id="d-live-old-claim",
    )
    ad._persist_completion(evt, {"summary": "SECRET RESULT"})
    assert ad.claim_completion_delivery("d-live-old-claim", "live-claim")

    with ad._transaction() as conn:
        conn.execute(
            "UPDATE async_delegations SET delivery_claimed_at = ? "
            "WHERE delegation_id = ?",
            (time.time() - 3600.0, "d-live-old-claim"),
        )

    assert not ad.claim_completion_delivery("d-live-old-claim", "thief-claim")
    assert ad.release_completion_delivery("d-live-old-claim", "live-claim")


def test_completion_ack_is_idempotent_by_delivery_identity(tmp_path, monkeypatch):
    import tools.async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "async_delegations.db")
    record = {
        "delegation_id": "d-idempotent-ack",
        "goal": "old goal",
        "context": None,
        "toolsets": None,
        "role": "leaf",
        "model": "m",
        "session_key": "OLD_SESSION_A",
        "origin_ui_session_id": "",
        "parent_session_id": "OLD_SESSION_A",
        "status": "running",
        "dispatched_at": time.time() - 2.0,
        "completed_at": None,
        "interrupt_fn": None,
    }
    ad._persist_dispatch(record)
    evt = _delegation_event(
        session_key="OLD_SESSION_A",
        delegation_id="d-idempotent-ack",
    )
    ad._persist_completion(evt, {"summary": "SECRET RESULT"})
    assert ad.claim_completion_delivery("d-idempotent-ack", "stable-claim")

    assert ad.complete_completion_delivery("d-idempotent-ack", "stable-claim")
    assert ad.complete_completion_delivery("d-idempotent-ack", "stable-claim")
    durable = ad.get_durable_delegation("d-idempotent-ack")
    assert durable is not None
    assert durable["delivery_state"] == "delivered"


def test_failed_delivery_attempts_still_stop_at_retry_bound(tmp_path, monkeypatch):
    import tools.async_delegation as ad

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "async_delegations.db")
    record = {
        "delegation_id": "d-retry-cap",
        "goal": "old goal",
        "context": None,
        "toolsets": None,
        "role": "leaf",
        "model": "m",
        "session_key": "OLD_SESSION_A",
        "origin_ui_session_id": "",
        "parent_session_id": "OLD_SESSION_A",
        "status": "running",
        "dispatched_at": time.time() - 2.0,
        "completed_at": None,
        "interrupt_fn": None,
    }
    ad._persist_dispatch(record)
    evt = _delegation_event(
        session_key="OLD_SESSION_A",
        delegation_id="d-retry-cap",
    )
    ad._persist_completion(evt, {"summary": "SECRET RESULT"})

    for attempt in range(ad._MAX_DELIVERY_ATTEMPTS):
        claim_id = f"failed-{attempt}"
        assert ad.claim_completion_delivery("d-retry-cap", claim_id)
        assert ad.release_completion_delivery("d-retry-cap", claim_id)

    durable = ad.get_durable_delegation("d-retry-cap")
    assert durable is not None
    assert durable["delivery_attempts"] == ad._MAX_DELIVERY_ATTEMPTS
    assert durable["delivery_state"] == "dropped"
    q = queue.Queue()
    assert ad.restore_undelivered_completions(q) == 0
    assert q.empty()


def test_owns_event_callback_beats_restored_flag():
    """A positive-proof ownership callback consumes restored events it owns."""
    reg = _make_registry()
    reg.completion_queue.put(_delegation_event(session_key="OWNER", restored=True))

    results = reg.drain_notifications(
        owns_event=lambda e: e.get("session_key") == "OWNER"
    )

    assert len(results) == 1
    assert reg.completion_queue.empty()
