"""Lifecycle-scoped gateway delivery regressions for terminal completions.

The gateway contract here is deliberately narrower than exactly-once: one live
GatewayRunner suppresses concurrent/replayed copies after successful adapter
injection, failed injection remains retryable, and durable async-delegation
state (when available) is acknowledged through its authoritative SQLite API.
"""

import asyncio
import contextlib
import json
import queue
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageDispatchStatus,
    MessageEvent,
    MessageType,
    SendResult,
    mark_message_consumed,
    resolve_message_acceptance,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Any current/future durable compatibility path must stay in tmp state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


def _runner(adapter, *, origins=None):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries=origins or {},
    )
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    return runner


def _accepting_adapter(*, failures: int = 0):
    """Adapter double that implements the durable acceptance protocol."""
    remaining_failures = failures

    async def _handle(event):
        nonlocal remaining_failures
        if remaining_failures:
            remaining_failures -= 1
            raise RuntimeError("temporary")
        if event.requires_durable_acceptance:
            resolve_message_acceptance(event, mark_message_consumed(event))
            return MessageDispatchStatus.ACCEPTED
        return None

    return SimpleNamespace(handle_message=AsyncMock(side_effect=_handle))


def _event_identity(event):
    """Return the stable inbox identity without abusing platform message IDs."""
    return event.message_id or (event.metadata or {}).get(
        "hermes_completion_delivery_id"
    )


class _RealQueueingAdapter(BasePlatformAdapter):
    """Minimal real adapter for exercising BasePlatformAdapter queue semantics."""

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict | None = None,
    ) -> SendResult:
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {}


class _LegacyNoneAdapter(_RealQueueingAdapter):
    """Third-party shape predating the durable dispatch return protocol."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.legacy_handle_calls = 0

    async def handle_message(self, event):
        self.legacy_handle_calls += 1
        return None


def _real_queueing_adapter():
    adapter = _RealQueueingAdapter(
        PlatformConfig(enabled=True, token="test", typing_indicator=False),
        Platform.TELEGRAM,
    )
    adapter._send_with_retry = AsyncMock(return_value=None)
    adapter._message_handler = AsyncMock(return_value=None)
    return adapter


def _async_event(delegation_id="deleg_duplicate"):
    return {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "agent:main:telegram:dm:12345:678",
        "goal": "Investigate flaky test",
        "status": "completed",
        "summary": "Found it",
        "api_calls": 1,
        "duration_seconds": 12.0,
        "dispatched_at": 1000.0,
        "completed_at": 1012.0,
        # PR #62479 stamps these on gateway-owned events. They must not
        # change the producer identity used for queue replay.
        "origin_profile": "default",
        "origin_hermes_home": "/tmp/hermes-default",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("arrival_order", ["user_first", "completion_first"])
async def test_busy_adapter_preserves_user_completion_identity_and_order_until_ack(
    isolated_registry,
    arrival_order,
):
    """Busy completion stays unacked in the bounded FIFO behind/ahead of user input."""
    from gateway.platforms.base import resolve_message_acceptance
    from tools import async_delegation

    event = _async_event("deleg_busy_false_ack")
    _persist_pending_completion(event)

    adapter = _real_queueing_adapter()
    runner = _runner(adapter)
    source = runner._build_process_event_source(event)
    assert source is not None
    session_key = build_session_key(source)
    queue_state = SimpleNamespace(conversation=SimpleNamespace(queued_events=[]))
    runner._adapter_for_source = lambda _source: adapter
    runner._session_state = lambda _session_key: queue_state
    runner._peek_session_state = lambda _session_key: queue_state
    runner._is_user_authorized = lambda _source: False  # internal event bypasses user auth
    adapter._busy_session_handler = runner._handle_active_session_busy_message

    foreground_release = asyncio.Event()

    async def _foreground_turn():
        await foreground_release.wait()

    foreground = asyncio.create_task(_foreground_turn())
    await asyncio.sleep(0)
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._session_tasks[session_key] = foreground

    user_pending = MessageEvent(
        text="keep this user follow-up intact",
        message_type=MessageType.TEXT,
        source=source,
        message_id="user-pending",
        internal=False,
    )
    delivery_task = None
    try:
        if arrival_order == "user_first":
            assert runner._queue_or_replace_pending_event(session_key, user_pending)
        delivery_task = asyncio.create_task(
            runner._deliver_completion_notification(
                "[async delegation completed]", event,
            )
        )
        for _ in range(20):
            await asyncio.sleep(0)
            pending_count = int(session_key in adapter._pending_messages) + len(
                queue_state.conversation.queued_events
            )
            if pending_count >= (1 if arrival_order == "completion_first" else 2):
                break
        if arrival_order == "completion_first":
            assert runner._queue_or_replace_pending_event(session_key, user_pending)

        ordered = []
        head = adapter._pending_messages.get(session_key)
        if head is not None:
            ordered.append(head)
        ordered.extend(queue_state.conversation.queued_events)
        expected_ids = (
            ["user-pending", "async-delegation:deleg_busy_false_ack"]
            if arrival_order == "user_first"
            else ["async-delegation:deleg_busy_false_ack", "user-pending"]
        )
        assert [_event_identity(queued) for queued in ordered] == expected_ids
        assert [queued.internal for queued in ordered] == (
            [False, True] if arrival_order == "user_first" else [True, False]
        )
        completion_event = next(queued for queued in ordered if queued.internal)
        assert completion_event.message_id is None
        assert completion_event.allow_gateway_control is False

        assert not delivery_task.done(), (
            delivery_task.result() if delivery_task.done() else None
        )
        durable = async_delegation.get_durable_delegation("deleg_busy_false_ack")
        assert durable is not None
        assert durable["delivery_state"] == "pending"
        assert durable["delivery_attempts"] == 1
        assert not foreground.done()
        adapter._message_handler.assert_not_awaited()

        # Only the exact queued completion's successful consumption receipt may
        # cross the durable ACK boundary.
        resolve_message_acceptance(completion_event, True)
        assert await delivery_task is True
        durable = async_delegation.get_durable_delegation("deleg_busy_false_ack")
        assert durable is not None
        assert durable["delivery_state"] == "delivered"
    finally:
        foreground_release.set()
        await foreground
        adapter._active_sessions.clear()
        adapter._session_tasks.clear()
        if delivery_task is not None and not delivery_task.done():
            delivery_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await delivery_task


@pytest.mark.asyncio
async def test_stop_then_retry_preserves_internal_user_identity_and_order(
    isolated_registry,
):
    """A /stop drains the queued completion without merging its identity."""
    from tools import async_delegation

    event = _async_event("deleg_stop_replay")
    _persist_pending_completion(event)
    adapter = _real_queueing_adapter()
    runner = _runner(adapter)
    source = runner._build_process_event_source(event)
    assert source is not None
    session_key = build_session_key(source)
    queue_state = SimpleNamespace(
        conversation=SimpleNamespace(queued_events=[]),
        turn=SimpleNamespace(agent=None),
        persistent=SimpleNamespace(pending_command_text=None),
    )
    runner._adapter_for_source = lambda _source: adapter
    runner._session_state = lambda _session_key: queue_state
    runner._peek_session_state = lambda _session_key: queue_state
    runner._is_user_authorized = lambda _source: False  # internal event bypasses user auth
    runner._invalidate_session_run_generation = lambda *_a, **_kw: 1
    adapter._busy_session_handler = runner._handle_active_session_busy_message

    async def _handle_event(handled_event):
        if handled_event.text == "/stop":
            await runner._interrupt_and_clear_session(
                session_key,
                source,
                interrupt_reason="test stop",
                invalidation_reason="test stop",
                release_running_state=False,
            )
        return None

    adapter._message_handler = AsyncMock(side_effect=_handle_event)

    foreground = asyncio.create_task(asyncio.Event().wait())
    await asyncio.sleep(0)
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._session_tasks[session_key] = foreground

    delivery_task = asyncio.create_task(
        runner._deliver_completion_notification(
            "[async delegation completed]", event,
        )
    )
    for _ in range(20):
        await asyncio.sleep(0)
        if session_key in adapter._pending_messages:
            break
    assert not delivery_task.done()
    assert not foreground.done()
    assert adapter._pending_messages[session_key].internal is True

    stop_event = MessageEvent(
        text="/stop",
        message_type=MessageType.TEXT,
        source=source,
        message_id="user-stop",
        internal=False,
    )
    await adapter.handle_message(stop_event)
    assert foreground.cancelled()

    # /stop discarded the volatile inbox copy. That first claim must fail
    # closed, leaving the durable completion replayable.
    assert await asyncio.wait_for(delivery_task, timeout=1.0) is not True
    durable = async_delegation.get_durable_delegation("deleg_stop_replay")
    assert durable is not None
    assert durable["delivery_state"] == "pending"

    # A fresh post-stop claim replays and acknowledges the completion once.
    assert await runner._deliver_completion_notification(
        "[async delegation completed]", event,
    ) is True
    durable = async_delegation.get_durable_delegation("deleg_stop_replay")
    assert durable is not None
    assert durable["delivery_state"] == "delivered"

    handled_events = [call.args[0] for call in adapter._message_handler.await_args_list]
    assert [handled.internal for handled in handled_events] == [False, True]
    assert handled_events[0] is stop_event
    assert handled_events[1].text == "[async delegation completed]"
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_repeated_stop_rejection_never_spends_delivery_attempt_budget(
    isolated_registry,
):
    """More than the failure cap of legitimate busy -> /stop cycles stays replayable."""
    from tools import async_delegation

    event = _async_event("deleg_many_stops")
    _persist_pending_completion(event)

    for _ in range(async_delegation._MAX_DELIVERY_ATTEMPTS + 2):
        adapter = _real_queueing_adapter()
        runner = _runner(adapter)
        source = runner._build_process_event_source(event)
        assert source is not None
        session_key = build_session_key(source)
        queue_state = SimpleNamespace(
            conversation=SimpleNamespace(queued_events=[]),
            turn=SimpleNamespace(agent=None),
            persistent=SimpleNamespace(pending_command_text=None),
        )
        runner._adapter_for_source = lambda _source, a=adapter: a
        runner._session_state = lambda _session_key, s=queue_state: s
        runner._peek_session_state = lambda _session_key, s=queue_state: s
        runner._is_user_authorized = lambda _source: False
        runner._invalidate_session_run_generation = lambda *_a, **_kw: 1
        adapter._busy_session_handler = runner._handle_active_session_busy_message

        foreground = asyncio.create_task(asyncio.Event().wait())
        await asyncio.sleep(0)
        adapter._active_sessions[session_key] = asyncio.Event()
        adapter._session_tasks[session_key] = foreground
        delivery_task = asyncio.create_task(
            runner._deliver_completion_notification("completion", event)
        )
        for _spin in range(100):
            await asyncio.sleep(0.001)
            if session_key in adapter._pending_messages:
                break
        assert session_key in adapter._pending_messages
        await runner._interrupt_and_clear_session(
            session_key,
            source,
            interrupt_reason="test stop",
            invalidation_reason="test stop",
            release_running_state=False,
        )
        assert await asyncio.wait_for(delivery_task, timeout=1) is not True
        row = async_delegation.get_durable_delegation("deleg_many_stops")
        assert row is not None
        assert row["delivery_state"] == "pending"
        assert row["delivery_attempts"] == 0
        foreground.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await foreground
        await adapter.cancel_background_tasks()

    accepting = _real_queueing_adapter()
    final_runner = _runner(accepting)
    assert await final_runner._deliver_completion_notification("completion", event) is True
    assert accepting._message_handler.await_count == 1
    await accepting.cancel_background_tasks()


@pytest.mark.asyncio
async def test_busy_completion_queue_cap_defers_without_ack_or_retry_charge(
    isolated_registry,
):
    """Bounded ownership rejects admission honestly instead of hanging/ACKing."""
    from tools import async_delegation

    event = _async_event("deleg_busy_cap")
    _persist_pending_completion(event)
    adapter = _real_queueing_adapter()
    runner = _runner(adapter)
    source = runner._build_process_event_source(event)
    assert source is not None
    session_key = build_session_key(source)
    queue_state = SimpleNamespace(conversation=SimpleNamespace(queued_events=[]))
    runner._adapter_for_source = lambda _source: adapter
    runner._session_state = lambda _session_key: queue_state
    runner._peek_session_state = lambda _session_key: queue_state
    adapter._busy_session_handler = runner._handle_active_session_busy_message

    foreground = asyncio.create_task(asyncio.Event().wait())
    await asyncio.sleep(0)
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._session_tasks[session_key] = foreground
    for index in range(runner._BUSY_QUEUE_MAX_PENDING):
        queued = MessageEvent(
            text=f"queued-{index}",
            message_type=MessageType.TEXT,
            source=source,
            message_id=f"user-{index}",
        )
        assert runner._queue_or_replace_pending_event(session_key, queued)

    try:
        disposition = await runner._deliver_completion_notification("completion", event)
        assert disposition is not True
        durable = async_delegation.get_durable_delegation("deleg_busy_cap")
        assert durable is not None
        assert durable["delivery_state"] == "pending"
        assert durable["delivery_attempts"] == 0
        all_pending = [adapter._pending_messages[session_key]] + list(
            queue_state.conversation.queued_events
        )
        assert len(all_pending) == runner._BUSY_QUEUE_MAX_PENDING
        assert all(not queued.internal for queued in all_pending)
    finally:
        foreground.cancel()
        with pytest.raises(asyncio.CancelledError):
            await foreground
        adapter._active_sessions.clear()
        adapter._session_tasks.clear()


@pytest.mark.asyncio
async def test_real_adapter_acknowledges_once_after_handler_acceptance(
    isolated_registry,
):
    from tools import async_delegation

    event = _async_event("deleg_real_exactly_once")
    _persist_pending_completion(event)
    adapter = _real_queueing_adapter()
    runner = _runner(adapter)

    first = await runner._deliver_completion_notification("completion", event)
    second = await runner._deliver_completion_notification("completion", event)

    assert first is True
    assert second is None
    assert adapter._message_handler.await_count == 1
    durable = async_delegation.get_durable_delegation("deleg_real_exactly_once")
    assert durable is not None
    assert durable["delivery_state"] == "delivered"
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_transcript_admission_closes_pre_marker_crash_gap_without_second_turn(
    isolated_registry,
):
    """A crash before the separate ledger marker dedupes from the transcript row."""
    from tools import async_delegation

    event = _async_event("deleg_consumed_ack_gap")
    _persist_pending_completion(event)
    runner = _runner(SimpleNamespace())
    runner._run_agent_inner = AsyncMock(side_effect=AssertionError("turn reran"))
    delivery_id = "async-delegation:deleg_consumed_ack_gap"
    history = [
        {
            "role": "user",
            "content": "[async delegation completed]",
            "message_id": delivery_id,
            "display_metadata": {
                "hermes_completion_delivery_ids": [delivery_id]
            },
        },
        {"role": "assistant", "content": "result incorporated"},
    ]

    # This is the exact pre-marker crash point: the transcript transaction
    # committed the turn, while the delegation ledger still has no consumed_at.
    durable = async_delegation.get_durable_delegation("deleg_consumed_ack_gap")
    assert durable is not None and durable["consumed_at"] is None

    result = await runner._run_agent(
        message="[async delegation completed]",
        context_prompt="",
        history=history,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat",
            user_id="user",
            chat_type="private",
        ),
        session_id="session",
        durable_delivery_ids=[delivery_id],
    )

    assert result["durable_delivery_replayed"] is True
    assert result["api_calls"] == 0
    assert result["messages"] == history
    runner._run_agent_inner.assert_not_awaited()
    assert sum(item.get("message_id") == delivery_id for item in history) == 1


@pytest.mark.asyncio
async def test_push_receipt_accepts_blank_persisted_terminal_proof():
    """A committed D,T transcript closes the push receipt after a pre-ACK crash."""
    from gateway.run import _should_clear_resume_pending_after_turn

    delivery_id = "async-delegation:deleg_push_pre_ack"
    history = [
        {
            "role": "user",
            "content": "completion",
            "message_id": delivery_id,
            "display_metadata": {
                "hermes_completion_delivery_ids": [delivery_id]
            },
        },
        {
            "role": "assistant",
            "content": "incorporated",
            "display_metadata": {
                "hermes_completion_delivery_ids": [delivery_id]
            },
        },
    ]
    runner = _runner(SimpleNamespace())
    runner._run_agent_inner = AsyncMock(side_effect=AssertionError("turn reran"))
    result = await runner._run_agent(
        message="completion",
        context_prompt="",
        history=history,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat",
            user_id="user",
            chat_type="private",
        ),
        session_id="session",
        durable_delivery_ids=[delivery_id],
    )
    history_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat",
        user_id="user",
        chat_type="private",
    )
    event = MessageEvent(
        text="completion",
        source=history_source,
        internal=True,
        requires_durable_acceptance=True,
    )
    consumed = MagicMock(return_value=True)
    setattr(event, "_hermes_mark_durable_consumed", consumed)
    if not _should_clear_resume_pending_after_turn(result):
        setattr(event, "_hermes_durable_acceptance_deferred", True)

    assert history_source == event.source
    assert result["final_response"] == ""
    assert result["durable_delivery_replayed"] is True
    assert mark_message_consumed(event) is True
    consumed.assert_called_once_with()


def test_push_receipt_uses_durable_parent_proof_when_failed_user_drains():
    """A failed fenced U cannot reject the already-terminal durable parent D."""
    from gateway.run import (
        _merge_queued_followup_result,
        _should_clear_resume_pending_after_turn,
    )

    durable_result = {
        "completed": True,
        "final_response": "durable done",
        "history_offset": 1,
    }
    failed_user_result = {
        "completed": False,
        "failed": True,
        "error": "later user failed",
        "final_response": "",
        "history_offset": 3,
    }
    drained_result = _merge_queued_followup_result(
        durable_result,
        failed_user_result,
        ["async-delegation:deleg_parent_then_failed_user"],
    )
    event = MessageEvent(
        text="completion",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat",
            user_id="user",
            chat_type="private",
        ),
        internal=True,
        requires_durable_acceptance=True,
    )
    consumed = MagicMock(return_value=True)
    setattr(event, "_hermes_mark_durable_consumed", consumed)
    if not _should_clear_resume_pending_after_turn(drained_result):
        setattr(event, "_hermes_durable_acceptance_deferred", True)

    assert drained_result["failed"] is True
    assert drained_result["error"] == "later user failed"
    assert drained_result["durable_parent_terminal_proof"] is True
    assert mark_message_consumed(event) is True
    consumed.assert_called_once_with()


@pytest.mark.asyncio
async def test_admitted_unfinished_completion_resumes_without_appending_user_turn(
    isolated_registry,
):
    """A durable user row without a terminal assistant resumes the same turn."""
    runner = _runner(SimpleNamespace())
    runner._run_agent_inner = AsyncMock(
        return_value={"completed": True, "messages": [], "final_response": "done"}
    )
    delivery_id = "async-delegation:deleg_resume_gap"
    history = [
        {
            "role": "user",
            "content": "completion",
            "message_id": delivery_id,
            "display_metadata": {
                "hermes_completion_delivery_ids": [delivery_id]
            },
        },
        {"role": "assistant", "content": None, "tool_calls": [{"id": "tc"}]},
        {"role": "tool", "content": "effect already persisted"},
    ]

    await runner._run_agent(
        message="completion",
        context_prompt="",
        history=history,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat",
            user_id="user",
            chat_type="private",
        ),
        session_id="session",
        durable_delivery_ids=[delivery_id],
    )

    kwargs = runner._run_agent_inner.await_args.kwargs
    assert kwargs["resume_admitted_turn"] is True
    assert kwargs["durable_delivery_ids"] == [delivery_id]


def test_durable_admission_terminal_proof_stops_at_next_user_turn():
    """A later turn's assistant can never prove the durable turn completed."""
    from gateway.durable_delivery import find_durable_admission

    delivery_id = "async-delegation:deleg_turn_boundary"
    history = [
        {
            "role": "user",
            "content": "durable completion",
            "message_id": delivery_id,
            "display_metadata": {
                "hermes_completion_delivery_ids": [delivery_id]
            },
        },
        {"role": "user", "content": "unrelated later request"},
        {"role": "assistant", "content": "unrelated later response"},
    ]

    admission_index, terminal = find_durable_admission(history, [delivery_id])

    assert admission_index == 0
    assert terminal is None


def test_unfinished_durable_turn_classifier_enforces_append_only_boundary():
    from gateway.durable_delivery import (
        UNFINISHED_DURABLE_CONFLICT,
        UNFINISHED_DURABLE_NONE,
        UNFINISHED_DURABLE_OPEN_TAIL,
        classify_unfinished_durable_turn,
    )

    delivery_id = "async-delegation:deleg_classifier"
    durable_user = {
        "role": "user",
        "content": "durable completion",
        "display_metadata": {"hermes_completion_delivery_ids": [delivery_id]},
    }
    tool_tail = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "tc"}]},
        {"role": "tool", "content": "effect persisted"},
    ]

    assert classify_unfinished_durable_turn([durable_user, *tool_tail]) == (
        UNFINISHED_DURABLE_OPEN_TAIL,
        [delivery_id],
        0,
    )
    assert classify_unfinished_durable_turn(
        [durable_user, *tool_tail, {"role": "user", "content": "later"}]
    ) == (UNFINISHED_DURABLE_CONFLICT, [delivery_id], 0)
    assert classify_unfinished_durable_turn(
        [durable_user, {"role": "assistant", "content": "legacy terminal"}]
    ) == (UNFINISHED_DURABLE_NONE, [], None)


def test_unfinished_durable_turn_fence_queues_new_user_but_allows_replay():
    adapter = SimpleNamespace(_pending_messages={})
    runner = _runner(adapter)

    def _park(session_key, event):
        adapter._pending_messages[session_key] = event
        return True

    runner._queue_or_replace_pending_event = MagicMock(side_effect=_park)
    runner._peek_session_state = lambda _session_key: None
    delivery_id = "async-delegation:deleg_fence"
    history = [
        {
            "role": "user",
            "content": "durable completion",
            "display_metadata": {
                "hermes_completion_delivery_ids": [delivery_id]
            },
        }
    ]
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat",
        user_id="user",
        chat_type="private",
    )
    later_user = MessageEvent(
        text="later request",
        message_type=MessageType.TEXT,
        source=source,
    )

    notice = runner._fence_unfinished_durable_turn(
        later_user, "session-key", history
    )

    assert "queued" in notice
    runner._queue_or_replace_pending_event.assert_called_once_with(
        "session-key", later_user
    )
    assert getattr(later_user, "_hermes_wait_for_durable_replay") is True

    durable_replay = MessageEvent(
        text="durable completion",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        metadata={"hermes_completion_delivery_ids": [delivery_id]},
    )
    assert (
        runner._fence_unfinished_durable_turn(
            durable_replay, "session-key", history
        )
        is None
    )
    # Admitting the replay does not release the parked user early. Only the
    # replay's terminal result opens the normal adapter drain.
    assert getattr(later_user, "_hermes_wait_for_durable_replay") is True
    runner._release_durable_fenced_events("session-key", source)
    assert not hasattr(later_user, "_hermes_wait_for_durable_replay")


def test_late_identity_linked_terminal_remains_append_only_conflict():
    """Identity cannot move a terminal backward across a later user boundary."""
    from gateway.durable_delivery import (
        UNFINISHED_DURABLE_CONFLICT,
        classify_unfinished_durable_turn,
        find_durable_admission,
    )

    delivery_id = "async-delegation:deleg_identity_terminal"
    terminal = {
        "role": "assistant",
        "content": "durable done",
        "display_metadata": {
            "hermes_completion_delivery_ids": [delivery_id]
        },
    }
    history = [
        {
            "role": "user",
            "content": "durable completion",
            "message_id": delivery_id,
            "display_metadata": {
                "hermes_completion_delivery_ids": [delivery_id]
            },
        },
        {"role": "user", "content": "unrelated later request"},
        {"role": "assistant", "content": "unrelated later response"},
        terminal,
    ]

    admission_index, found = find_durable_admission(history, [delivery_id])

    assert admission_index == 0
    assert found is None
    assert classify_unfinished_durable_turn(history) == (
        UNFINISHED_DURABLE_CONFLICT,
        [delivery_id],
        0,
    )


@pytest.mark.asyncio
async def test_admitted_turn_resume_isolated_from_later_user_turn():
    """Recovery fails closed once a later user boundary exists."""
    runner = _runner(SimpleNamespace())
    delivery_id = "async-delegation:deleg_isolated_resume"
    durable_user = {
        "role": "user",
        "content": "durable completion",
        "message_id": delivery_id,
        "display_metadata": {
            "hermes_completion_delivery_ids": [delivery_id]
        },
    }
    later_turn = [
        {"role": "user", "content": "unrelated later request"},
        {"role": "assistant", "content": "unrelated later response"},
    ]

    runner._run_agent_inner = AsyncMock()
    result = await runner._run_agent(
        message="durable completion",
        context_prompt="",
        history=[durable_user, *later_turn],
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat",
            user_id="user",
            chat_type="private",
        ),
        session_id="session",
        durable_delivery_ids=[delivery_id],
    )

    assert result["messages"] == [durable_user, *later_turn]
    assert result["durable_turn_conflict"] is True
    assert result["retryable"] is True
    runner._run_agent_inner.assert_not_awaited()


@pytest.mark.asyncio
async def test_admitted_turn_resume_keeps_provider_prefix_and_reload_order(
    tmp_path,
):
    """A resumed old receipt must not exist in two different transcript orders."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    delivery_id = "async-delegation:deleg_persisted_resume_order"
    db = SessionDB(db_path=tmp_path / "resume-order.db")
    session_id = "persisted-resume-order"
    db.create_session(session_id, source="telegram")
    db.append_message(
        session_id,
        "user",
        "durable completion",
        display_metadata={"hermes_completion_delivery_ids": [delivery_id]},
    )
    db.append_message(session_id, "user", "unrelated later request")
    db.append_message(session_id, "assistant", "unrelated later response")
    history = db.get_messages_as_conversation(session_id)
    provider_visible_prefix = [
        (message["role"], message.get("content")) for message in history
    ]

    # Use the production incremental writer.  The gateway's inner runner owns
    # the fully configured agent in production; only persistence state is
    # needed to reproduce this boundary here.
    agent = object.__new__(AIAgent)
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._db_flush_scan_prefix = None
    observed_call_history = []

    async def _resume(_message, _context, call_history, _source, _session, **kwargs):
        observed_call_history.extend(
            (message["role"], message.get("content")) for message in call_history
        )
        terminal = {
            "role": "assistant",
            "content": "durable done",
            "display_metadata": {
                "hermes_completion_delivery_ids": [delivery_id]
            },
        }
        messages = [*call_history, terminal]
        assert agent._flush_messages_to_session_db(
            messages, conversation_history=call_history
        ) is True
        return {
            "completed": True,
            "messages": messages,
            "final_response": "durable done",
        }

    runner = _runner(SimpleNamespace())
    runner._run_agent_inner = AsyncMock(side_effect=_resume)
    result = await runner._run_agent(
        message="durable completion",
        context_prompt="",
        history=history,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat",
            user_id="user",
            chat_type="private",
        ),
        session_id=session_id,
        durable_delivery_ids=[delivery_id],
    )

    live_order = [
        (message["role"], message.get("content")) for message in result["messages"]
    ]
    reloaded_order = [
        (message["role"], message.get("content"))
        for message in db.get_messages_as_conversation(session_id)
    ]
    assert result["durable_turn_conflict"] is True
    assert result["retryable"] is True
    assert observed_call_history == []
    assert reloaded_order == live_order == provider_visible_prefix


@pytest.mark.asyncio
async def test_unresolved_durable_receipt_is_bounded_and_deferred(
    isolated_registry,
):
    """An ACCEPTED adapter that forgets its receipt cannot hold a claim forever."""
    from tools import async_delegation

    event = _async_event("deleg_unresolved_receipt")
    _persist_pending_completion(event)
    adapter = SimpleNamespace(
        handle_message=AsyncMock(return_value=MessageDispatchStatus.ACCEPTED)
    )
    runner = _runner(adapter)
    runner._durable_acceptance_timeout_seconds = 0.01

    result = await asyncio.wait_for(
        runner._deliver_completion_notification("completion", event),
        timeout=0.5,
    )

    assert result is not True
    durable = async_delegation.get_durable_delegation("deleg_unresolved_receipt")
    assert durable is not None
    assert durable["delivery_state"] == "pending"
    assert durable["delivery_attempts"] == 0


@pytest.mark.asyncio
async def test_handler_failure_releases_durable_completion_for_retry(
    isolated_registry,
):
    from tools import async_delegation

    event = _async_event("deleg_handler_retry")
    _persist_pending_completion(event)
    adapter = _real_queueing_adapter()
    adapter._message_handler = AsyncMock(
        side_effect=[RuntimeError("handler failed before acceptance"), None]
    )
    runner = _runner(adapter)

    first = await runner._deliver_completion_notification("completion", event)
    pending = async_delegation.get_durable_delegation("deleg_handler_retry")
    second = await runner._deliver_completion_notification("completion", event)
    delivered = async_delegation.get_durable_delegation("deleg_handler_retry")

    assert first is False
    assert pending is not None
    assert pending["delivery_state"] == "pending"
    assert pending["delivery_attempts"] == 1
    assert second is True
    assert delivered is not None
    assert delivered["delivery_state"] == "delivered"
    assert adapter._message_handler.await_count == 2
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_rejected_durable_ack_does_not_emit_false_delivered_state(
    isolated_registry,
    monkeypatch,
):
    """Handler return is insufficient when the authoritative ACK CAS rejects."""
    from tools import async_delegation

    event = _async_event("deleg_ack_rejected")
    _persist_pending_completion(event)
    adapter = _real_queueing_adapter()
    runner = _runner(adapter)
    monkeypatch.setattr(async_delegation, "complete_completion_delivery", lambda *_: False)

    assert await runner._deliver_completion_notification("completion", event) is False

    durable = async_delegation.get_durable_delegation("deleg_ack_rejected")
    assert durable is not None
    assert durable["delivery_state"] == "pending"
    assert durable["delivery_attempts"] == 1
    identity = runner._completion_delivery_identity(event)
    assert identity not in runner._completion_deliveries_delivered
    assert identity not in runner._completion_deliveries_inflight
    await adapter.cancel_background_tasks()


def _completion_event(*, started_at, session_id="proc_reused"):
    return {
        "type": "completion",
        "session_id": session_id,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "started_at": started_at,
        "command": "echo done",
        "exit_code": 0,
        "completion_reason": "exited",
        "output": "done\n",
    }


def _stop_after_sleeps(monkeypatch, runner, count):
    sleep_calls = 0

    async def _bounded_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= count:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", _bounded_sleep)


def test_duplicate_async_queue_replay_injects_once(monkeypatch, isolated_registry):
    """Byte-identical queue replays produce one turn in one gateway lifecycle."""
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(dict(_async_event()))
    isolated.put(dict(_async_event()))

    adapter = _accepting_adapter()
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()


def test_unroutable_async_event_is_not_requeued_forever(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    event = _async_event("deleg_desktop_or_cli")
    event["session_key"] = "20260711_unparseable_ui_session"
    isolated.put(event)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_not_awaited()
    assert isolated.empty()


def test_concurrent_claims_share_the_same_narrow_delivery_seam():
    """Concurrent consumers in one runner cannot both enter the adapter."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _blocked_injection(event):
        entered.set()
        await release.wait()
        resolve_message_acceptance(event, True)
        return MessageDispatchStatus.ACCEPTED

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_blocked_injection))
    runner = _runner(adapter)
    event = _async_event()
    text = "completion"

    async def _exercise():
        first = asyncio.create_task(runner._deliver_completion_notification(text, dict(event)))
        await entered.wait()
        second = asyncio.create_task(runner._deliver_completion_notification(text, dict(event)))
        await asyncio.sleep(0)
        release.set()
        return await asyncio.gather(first, second)

    assert sorted(asyncio.run(_exercise()), key=str) == [None, True]
    adapter.handle_message.assert_awaited_once()


def test_failed_async_injection_is_retried_and_only_success_is_acked(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(_async_event())

    adapter = _accepting_adapter(failures=1)
    runner = _runner(adapter)
    sleep_calls = 0
    real_sleep = asyncio.sleep

    async def _stop_after_retry(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if adapter.handle_message.await_count >= 2 and isolated.empty():
            runner._running = False
        elif sleep_calls >= 2000:
            runner._running = False
        await real_sleep(0.001)

    monkeypatch.setattr(asyncio, "sleep", _stop_after_retry)

    from tools import async_delegation

    acknowledgements = []
    monkeypatch.setattr(
        async_delegation,
        "complete_completion_delivery",
        lambda delegation_id, _claim_id: acknowledgements.append(delegation_id) or True,
        raising=False,
    )

    asyncio.run(runner._async_delegation_watcher(interval=0))

    assert adapter.handle_message.await_count == 2
    assert acknowledgements == ["deleg_duplicate"]


def test_durable_injection_rejects_legacy_bare_none_without_false_ack():
    """Legacy adapter ambiguity fails closed for every durable producer."""
    from tools import async_delegation

    event = _async_event("deleg_legacy_none")
    _persist_pending_completion(event)
    adapter = SimpleNamespace(handle_message=AsyncMock(return_value=None))
    runner = _runner(adapter)

    assert asyncio.run(
        runner._deliver_completion_notification("completion", event)
    ) is False
    durable = async_delegation.get_durable_delegation("deleg_legacy_none")
    assert durable is not None
    assert durable["delivery_state"] == "pending"


def test_durable_injection_rejects_missing_persisted_origin_profile():
    """Profile-less durable rows are malformed, never silently routed to default."""
    from tools import async_delegation

    event = _async_event("deleg_missing_profile")
    event.pop("origin_profile")
    _persist_pending_completion(event)
    adapter = SimpleNamespace(handle_message=AsyncMock(return_value=None))
    runner = _runner(adapter)

    assert asyncio.run(
        runner._deliver_completion_notification("completion", event)
    ) is False
    adapter.handle_message.assert_not_awaited()
    durable = async_delegation.get_durable_delegation("deleg_missing_profile")
    assert durable is not None
    assert durable["delivery_state"] == "pending"
    identity = runner._completion_delivery_identity(event)
    assert identity not in runner._completion_deliveries_delivered


@pytest.mark.asyncio
async def test_api_self_post_receives_identity_built_before_transport_branch(monkeypatch):
    event = _async_event("deleg_api_self_post")
    event["session_key"] = "agent:main:api_server:dm:raw-api-session"
    adapter = SimpleNamespace(supports_async_delivery=False)
    runner = _runner(adapter)
    runner.adapters = {Platform.API_SERVER: adapter}
    delivered = AsyncMock(return_value=None)

    monkeypatch.setattr("gateway.wake.deliver_wake", delivered)

    assert await runner._inject_watch_notification("completion", event) is True
    assert delivered.await_args.kwargs["session_id"] == "raw-api-session"
    assert delivered.await_args.kwargs["durable_delivery_ids"] == [
        "async-delegation:deleg_api_self_post"
    ]


@pytest.mark.asyncio
async def test_base_owned_durable_dispatch_bypasses_legacy_none_override_once():
    """A Base subclass's historical ``None`` override cannot swallow inbox work."""
    from tools import async_delegation

    event = _async_event("deleg_legacy_base_dispatch")
    _persist_pending_completion(event)
    adapter = _LegacyNoneAdapter(
        PlatformConfig(enabled=True, token="test", typing_indicator=False),
        Platform.TELEGRAM,
    )
    consumed = 0

    async def _consume(_event):
        nonlocal consumed
        consumed += 1
        return None

    adapter._message_handler = _consume
    runner = _runner(adapter)

    assert await runner._deliver_completion_notification("completion", event) is True
    row = async_delegation.get_durable_delegation("deleg_legacy_base_dispatch")
    assert row is not None
    assert row["delivery_state"] == "delivered"
    assert consumed == 1
    assert adapter.legacy_handle_calls == 0
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_acceptance_timeout_retains_owner_until_slow_handler_eventually_acks():
    """A liveness timeout must not expose a live admitted turn for replay."""
    from tools import async_delegation

    event = _async_event("deleg_slow_acceptance")
    _persist_pending_completion(event)
    adapter = _real_queueing_adapter()
    started = asyncio.Event()
    release = asyncio.Event()
    invocations = 0

    async def _slow_handler(_event):
        nonlocal invocations
        invocations += 1
        started.set()
        await release.wait()

    adapter._message_handler = _slow_handler
    runner = _runner(adapter)
    runner._durable_acceptance_timeout_seconds = 0.01

    delivery = asyncio.create_task(
        runner._deliver_completion_notification("completion", event)
    )
    await started.wait()
    await asyncio.sleep(0.03)

    assert delivery.done() is False
    assert async_delegation.claim_event_delivery(
        event, consumer="concurrent-replay"
    ) is None
    assert invocations == 1

    release.set()
    assert await delivery is True
    row = async_delegation.get_durable_delegation("deleg_slow_acceptance")
    assert row is not None
    assert row["delivery_state"] == "delivered"
    assert invocations == 1
    await adapter.cancel_background_tasks()


def test_incomplete_remote_result_cannot_run_durable_consumption_hook():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm")
    event = MessageEvent(
        text="completion",
        source=source,
        internal=True,
        requires_durable_acceptance=True,
    )
    consumed = MagicMock(return_value=True)
    setattr(event, "_hermes_mark_durable_consumed", consumed)
    setattr(event, "_hermes_durable_acceptance_deferred", True)

    assert mark_message_consumed(event) is False
    consumed.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_delivery_cancels_base_owner_and_leaves_row_replayable():
    """Cancellation unwinds the admitted task before releasing its durable claim."""
    from tools import async_delegation

    event = _async_event("deleg_cancelled_acceptance")
    _persist_pending_completion(event)
    adapter = _real_queueing_adapter()
    started = asyncio.Event()
    owner_cancelled = asyncio.Event()

    async def _blocked_handler(_event):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            owner_cancelled.set()
            raise

    adapter._message_handler = _blocked_handler
    runner = _runner(adapter)
    delivery = asyncio.create_task(
        runner._deliver_completion_notification("completion", event)
    )
    await started.wait()
    delivery.cancel()
    with pytest.raises(asyncio.CancelledError):
        await delivery

    await asyncio.wait_for(owner_cancelled.wait(), timeout=1.0)
    replay_claim = async_delegation.claim_event_delivery(
        event, consumer="restart-replay"
    )
    assert replay_claim
    async_delegation.defer_completion_delivery(
        "deleg_cancelled_acceptance", replay_claim
    )
    row = async_delegation.get_durable_delegation("deleg_cancelled_acceptance")
    assert row is not None
    assert row["delivery_state"] == "pending"
    assert row["consumed_at"] is None
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_adapter_shutdown_rejects_durable_queue_without_spooling(tmp_path):
    """Shutdown drops only the volatile copy and settles its receipt for replay."""
    adapter = _real_queueing_adapter()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="shutdown-chat",
        chat_type="dm",
    )
    event = MessageEvent(
        text="durable completion",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        requires_durable_acceptance=True,
    )
    receipt = asyncio.get_running_loop().create_future()
    setattr(event, "_hermes_durable_acceptance_receipt", receipt)
    session_key = build_session_key(source)
    adapter._pending_messages[session_key] = event

    await adapter.cancel_background_tasks()

    assert receipt.result() is False
    assert adapter._pending_messages == {}
    assert not list((tmp_path / "pending_messages").glob("*.json"))


@pytest.mark.asyncio
async def test_adapter_shutdown_defers_durable_claim_without_charging(
    isolated_registry,
):
    """Gateway shutdown is backpressure, not a failed delivery attempt."""
    from tools import async_delegation

    durable_event = _async_event("deleg_shutdown_defer")
    _persist_pending_completion(durable_event)
    adapter = _real_queueing_adapter()
    runner = _runner(adapter)
    source = runner._build_process_event_source(durable_event)
    assert source is not None
    session_key = build_session_key(source)
    state = SimpleNamespace(conversation=SimpleNamespace(queued_events=[]))
    runner._adapter_for_source = lambda _source: adapter
    runner._session_state = lambda _session_key: state
    runner._peek_session_state = lambda _session_key: state
    runner._is_user_authorized = lambda _source: False
    adapter._busy_session_handler = runner._handle_active_session_busy_message
    adapter._active_sessions[session_key] = asyncio.Event()
    foreground = asyncio.create_task(asyncio.Event().wait())
    adapter._session_tasks[session_key] = foreground

    delivery = asyncio.create_task(
        runner._deliver_completion_notification("completion", durable_event)
    )
    for _ in range(100):
        await asyncio.sleep(0.001)
        if session_key in adapter._pending_messages:
            break
    assert session_key in adapter._pending_messages

    await adapter.cancel_background_tasks()

    assert await asyncio.wait_for(delivery, timeout=1) is not True
    row = async_delegation.get_durable_delegation("deleg_shutdown_defer")
    assert row is not None
    assert row["delivery_state"] == "pending"
    assert row["delivery_attempts"] == 0


def _persist_pending_completion(event):
    from tools import async_delegation

    async_delegation._persist_dispatch({
        "delegation_id": event["delegation_id"],
        "session_key": event["session_key"],
        "origin_ui_session_id": "",
        "parent_session_id": event.get("parent_session_id"),
        "dispatched_at": event["dispatched_at"],
    })
    async_delegation._persist_completion(event, {
        "status": "completed",
        "summary": event["summary"],
    })


def test_explicit_kill_returns_output_before_consuming_notification(monkeypatch):
    import tools.process_registry as pr_module

    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_kill_consumed",
        command="sleep 999",
        task_id="task",
        started_at=1.0,
        output_buffer="important terminal output\n",
        notify_on_complete=True,
    )
    session.process = MagicMock()
    session.process.pid = 4242
    registry._running[session.id] = session
    monkeypatch.setattr(registry, "_terminate_host_pid", lambda *_a, **_kw: None)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(pr_module, "process_registry", registry)

    result = registry.kill_process(session.id)
    assert result["status"] == "killed"
    assert result["output"] == "important terminal output\n"
    assert registry.is_completion_consumed(session.id)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    asyncio.run(runner._run_process_watcher({
        "session_id": session.id,
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }))

    adapter.handle_message.assert_not_awaited()


def test_process_tool_redacts_explicit_kill_output(monkeypatch):
    from tools import process_registry as pr_module

    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_kill_redacted",
        command="printenv",
        task_id="task",
        started_at=1.0,
        output_buffer="PRIVATE_TOKEN=opaque-value\n",
        exited=True,
        exit_code=0,
    )
    registry._finished[session.id] = session
    monkeypatch.setattr(pr_module, "process_registry", registry)

    def _redact(result):
        assert result["output"] == "PRIVATE_TOKEN=opaque-value\n"
        result["output"] = "PRIVATE_TOKEN=<redacted>\n"
        return result

    monkeypatch.setattr(pr_module, "_redact_process_result", _redact)

    result = json.loads(pr_module._handle_process({
        "action": "kill",
        "session_id": session.id,
    }))
    assert result["output"] == "PRIVATE_TOKEN=<redacted>\n"


def test_autonomous_completion_redacts_real_command_and_output_secrets(monkeypatch):
    import agent.redact as redact_module
    import tools.process_registry as pr_module

    secret = "abc123randomopaquetokenvalue999"
    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_autonomous_redaction",
        command=f"printenv MY_SERVICE_TOKEN={secret}",
        task_id="task",
        started_at=1234.5,
        output_buffer=f"MY_SERVICE_TOKEN={secret}\nHOME=/home/user\n",
        exited=True,
        exit_code=0,
        notify_on_complete=True,
    )
    registry._finished[session.id] = session
    monkeypatch.setattr(pr_module, "process_registry", registry)
    monkeypatch.setattr(redact_module, "_REDACT_ENABLED", True)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    asyncio.run(runner._run_process_watcher({
        "session_id": session.id,
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }))

    delivered = adapter.handle_message.await_args.args[0]
    assert secret not in delivered.text
    assert "HOME=/home/user" in delivered.text


def test_concurrent_process_watchers_coalesce_one_session_completion_turn(monkeypatch):
    """Concurrent terminal watchers for one session must re-enter the agent once."""
    import tools.process_registry as pr_module

    registry = ProcessRegistry()
    watchers = []
    for index in range(3):
        session = ProcessSession(
            id=f"proc_batch_{index}",
            command=f"printf batch-{index}",
            task_id=f"task-{index}",
            started_at=1000.0 + index,
            output_buffer=f"batch-{index}\n",
            exited=True,
            exit_code=0,
            notify_on_complete=True,
        )
        registry._finished[session.id] = session
        watchers.append({
            "session_id": session.id,
            "check_interval": 0,
            "session_key": "agent:main:telegram:dm:123",
            "platform": "telegram",
            "chat_type": "dm",
            "chat_id": "123",
            "notify_on_complete": True,
        })
    monkeypatch.setattr(pr_module, "process_registry", registry)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _exercise():
        await asyncio.gather(*(
            runner._run_process_watcher(watcher)
            for watcher in watchers
        ))

    asyncio.run(_exercise())

    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    assert "3 background processes completed" in delivered.text
    for index in range(3):
        assert f"proc_batch_{index}" in delivered.text


def test_completion_arriving_during_batch_delivery_schedules_next_flush():
    """A new event cannot be stranded behind an in-flight batch for its route."""
    first_delivery_entered = asyncio.Event()
    release_first_delivery = asyncio.Event()
    delivery_count = 0

    async def _deliver(_event):
        nonlocal delivery_count
        delivery_count += 1
        if delivery_count == 1:
            first_delivery_entered.set()
            await release_first_delivery.wait()

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_deliver))
    runner = _runner(adapter)

    async def _exercise():
        first = asyncio.create_task(runner._enqueue_process_completion_notification(
            "first completion",
            _completion_event(started_at=1.0, session_id="proc_first"),
        ))
        await first_delivery_entered.wait()
        second = asyncio.create_task(runner._enqueue_process_completion_notification(
            "second completion",
            _completion_event(started_at=2.0, session_id="proc_second"),
        ))
        release_first_delivery.set()
        assert await first is True
        assert await asyncio.wait_for(second, timeout=1.0) is True

    asyncio.run(_exercise())

    assert adapter.handle_message.await_count == 2


def test_completion_batches_do_not_cross_conversation_routes():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    first = _completion_event(started_at=1.0, session_id="proc_route_a")
    second = _completion_event(started_at=2.0, session_id="proc_route_b")
    second["session_key"] = "agent:main:telegram:dm:456"
    second["chat_id"] = "456"

    async def _exercise():
        return await asyncio.gather(
            runner._enqueue_process_completion_notification("first", first),
            runner._enqueue_process_completion_notification("second", second),
        )

    assert asyncio.run(_exercise()) == [True, True]
    assert adapter.handle_message.await_count == 2


def test_failed_coalesced_delivery_retries_all_entries():
    attempts = 0

    async def _deliver(_event):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary adapter failure")

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_deliver))
    runner = _runner(adapter)
    events = [
        _completion_event(started_at=float(index), session_id=f"proc_retry_{index}")
        for index in range(2)
    ]

    async def _enqueue_all():
        return await asyncio.gather(*(
            runner._enqueue_process_completion_notification(f"event-{index}", event)
            for index, event in enumerate(events)
        ))

    async def _exercise():
        assert await _enqueue_all() == [False, False]
        assert await _enqueue_all() == [True, True]

    asyncio.run(_exercise())
    assert adapter.handle_message.await_count == 2


def test_coalesced_success_records_every_completion_identity():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    events = [
        _completion_event(started_at=float(index), session_id=f"proc_ledger_{index}")
        for index in range(3)
    ]

    async def _exercise():
        return await asyncio.gather(*(
            runner._enqueue_process_completion_notification(f"event-{index}", event)
            for index, event in enumerate(events)
        ))

    assert asyncio.run(_exercise()) == [True, True, True]
    for event in events:
        identity = runner._completion_delivery_identity(event)
        assert identity in runner._completion_deliveries_delivered


def test_coalesced_format_bounds_details_and_reports_omitted_count():
    async def _format():
        loop = asyncio.get_running_loop()
        entries = [
            (
                f"event-{index}",
                _completion_event(
                    started_at=float(index), session_id=f"proc_bound_{index}"
                ),
                loop.create_future(),
            )
            for index in range(12)
        ]
        return GatewayRunner._format_coalesced_process_completions(entries)

    text = asyncio.run(_format())

    for index in range(10):
        assert f"proc_bound_{index}" in text
    assert "proc_bound_10" not in text
    assert "proc_bound_11" not in text
    assert "and 2 more completion(s)" in text


def test_coalesced_format_force_redacts_output_when_redaction_disabled(monkeypatch):
    """A user setting cannot disable the gateway's outbound secret floor."""
    import agent.redact as redact_module

    secret = "abc123randomopaquetokenvalue999"
    monkeypatch.setattr(redact_module, "_REDACT_ENABLED", False)

    async def _format():
        loop = asyncio.get_running_loop()
        first = _completion_event(started_at=1.0, session_id="proc_secret")
        first["output"] = (
            f"MY_SERVICE_TOKEN={secret}\n"
            "HOME=/home/user\n"
        )
        second = _completion_event(started_at=2.0, session_id="proc_control")
        return GatewayRunner._format_coalesced_process_completions([
            ("first", first, loop.create_future()),
            ("second", second, loop.create_future()),
        ])

    text = asyncio.run(_format())

    assert secret not in text
    assert "HOME=/home/user" in text


def test_coalesced_format_redacts_before_truncating_output(monkeypatch):
    """Truncation cannot remove the prefix needed to recognize a secret."""
    import agent.redact as redact_module

    marker = "SHOULD_NOT_SURVIVE"
    monkeypatch.setattr(redact_module, "_REDACT_ENABLED", False)

    async def _format():
        loop = asyncio.get_running_loop()
        first = _completion_event(started_at=1.0, session_id="proc_long_secret")
        first["output"] = f"MY_SERVICE_TOKEN={'x' * 900}{marker}\n"
        second = _completion_event(started_at=2.0, session_id="proc_control")
        return GatewayRunner._format_coalesced_process_completions([
            ("first", first, loop.create_future()),
            ("second", second, loop.create_future()),
        ])

    text = asyncio.run(_format())

    assert marker not in text


def test_duplicate_primary_does_not_discard_fresh_batch_sibling():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    duplicate = _completion_event(started_at=1.0, session_id="proc_duplicate")
    fresh = _completion_event(started_at=2.0, session_id="proc_fresh")
    duplicate_identity = runner._completion_delivery_identity(duplicate)
    runner._completion_deliveries_delivered[duplicate_identity] = None

    async def _exercise():
        return await asyncio.gather(
            runner._enqueue_process_completion_notification("duplicate", duplicate),
            runner._enqueue_process_completion_notification("fresh", fresh),
        )

    assert asyncio.run(_exercise()) == [True, True]
    adapter.handle_message.assert_awaited_once()
    fresh_identity = runner._completion_delivery_identity(fresh)
    assert fresh_identity in runner._completion_deliveries_delivered


def test_batch_format_failure_resolves_waiters_for_retry(monkeypatch):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    monkeypatch.setattr(
        runner,
        "_format_coalesced_process_completions",
        MagicMock(side_effect=ValueError("bad batch")),
    )
    events = [
        _completion_event(started_at=float(index), session_id=f"proc_format_{index}")
        for index in range(2)
    ]

    async def _exercise():
        pending = asyncio.gather(*(
            runner._enqueue_process_completion_notification(f"event-{index}", event)
            for index, event in enumerate(events)
        ))
        return await asyncio.wait_for(pending, timeout=1.0)

    assert asyncio.run(_exercise()) == [False, False]
    adapter.handle_message.assert_not_awaited()


def test_shutdown_cancels_batch_during_window_and_settles_waiter_for_retry():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    sleep_entered = asyncio.Event()
    release_sleep = asyncio.Event()
    real_sleep = asyncio.sleep
    event = _completion_event(started_at=1.0, session_id="proc_cancel_window")

    async def _controlled_sleep(delay):
        if delay == runner._completion_notification_batch_window:
            sleep_entered.set()
            await release_sleep.wait()
            return
        await real_sleep(delay)

    async def _exercise():
        pending = asyncio.create_task(
            runner._enqueue_process_completion_notification("completion", event)
        )
        await sleep_entered.wait()
        flush_task = next(iter(runner._completion_notification_batch_tasks.values()))
        assert flush_task in runner._background_tasks

        await runner._cancel_process_completion_batch_tasks()

        assert await asyncio.wait_for(pending, timeout=1.0) is False
        assert flush_task.cancelled()
        assert flush_task not in runner._background_tasks
        assert runner._completion_notification_batches == {}
        assert runner._completion_notification_batch_tasks == {}

    with patch("gateway.run.asyncio.sleep", new=_controlled_sleep):
        asyncio.run(_exercise())
    adapter.handle_message.assert_not_awaited()


def test_shutdown_cancels_blocked_batch_delivery_and_keeps_it_retryable():
    delivery_entered = asyncio.Event()

    async def _blocked_delivery(_event):
        delivery_entered.set()
        await asyncio.Event().wait()

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_blocked_delivery))
    runner = _runner(adapter)
    runner._completion_notification_batch_window = 0
    event = _completion_event(started_at=1.0, session_id="proc_cancel_delivery")

    async def _exercise():
        pending = asyncio.create_task(
            runner._enqueue_process_completion_notification("completion", event)
        )
        await delivery_entered.wait()
        flush_task = next(iter(runner._completion_notification_batch_flush_tasks))

        await runner._cancel_process_completion_batch_tasks()

        assert await asyncio.wait_for(pending, timeout=1.0) is False
        assert flush_task.cancelled()
        assert runner._completion_delivery_identity(event) not in runner._completion_deliveries_inflight
        assert runner._completion_delivery_identity(event) not in runner._completion_deliveries_delivered
        assert runner._completion_notification_batches == {}
        assert runner._completion_notification_batch_tasks == {}

    asyncio.run(_exercise())
    adapter.handle_message.assert_awaited_once()


def test_completion_enqueue_stays_retryable_after_shutdown_starts():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _exercise():
        await runner._cancel_process_completion_batch_tasks()
        return await runner._enqueue_process_completion_notification(
            "completion",
            _completion_event(started_at=1.0, session_id="proc_after_shutdown"),
        )

    assert asyncio.run(_exercise()) is False
    assert runner._completion_notification_batches == {}
    assert runner._completion_notification_batch_tasks == {}
    adapter.handle_message.assert_not_awaited()


def test_successful_batch_releases_all_lifecycle_task_references():
    adapter = SimpleNamespace(handle_message=AsyncMock(return_value=None))
    runner = _runner(adapter)
    runner._completion_notification_batch_window = 0

    async def _exercise():
        result = await runner._enqueue_process_completion_notification(
            "completion",
            _completion_event(started_at=1.0, session_id="proc_success_cleanup"),
        )
        await asyncio.sleep(0)
        return result

    assert asyncio.run(_exercise()) is True
    assert runner._completion_notification_batch_tasks == {}
    assert runner._completion_notification_batch_flush_tasks == set()
    assert runner._background_tasks == set()


def test_shutdown_cancels_overlapping_flushes_for_same_route():
    delivery_entered = asyncio.Event()

    async def _blocked_delivery(_event):
        delivery_entered.set()
        await asyncio.Event().wait()

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_blocked_delivery))
    runner = _runner(adapter)
    runner._completion_notification_batch_window = 0
    first_event = _completion_event(started_at=1.0, session_id="proc_old_flush")
    second_event = _completion_event(started_at=2.0, session_id="proc_new_flush")

    async def _exercise():
        first = asyncio.create_task(
            runner._enqueue_process_completion_notification("first", first_event)
        )
        await delivery_entered.wait()

        # The first task has detached from the route index while blocked in
        # adapter delivery.  A new completion for the same route must create a
        # second flush, and shutdown must still own and cancel both tasks.
        assert runner._completion_notification_batch_tasks == {}
        runner._completion_notification_batch_window = 3600
        second = asyncio.create_task(
            runner._enqueue_process_completion_notification("second", second_event)
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        flush_tasks = set(runner._completion_notification_batch_flush_tasks)
        assert len(flush_tasks) == 2

        await runner._cancel_process_completion_batch_tasks()

        assert await asyncio.gather(first, second) == [False, False]
        assert all(task.cancelled() for task in flush_tasks)
        assert runner._completion_notification_batches == {}
        assert runner._completion_notification_batch_tasks == {}
        assert runner._completion_notification_batch_flush_tasks == set()
        assert runner._background_tasks == set()

    asyncio.run(_exercise())
    adapter.handle_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# Async-delegation same-tick coalescing (#70300)
# ---------------------------------------------------------------------------


def _distinct_async_event(delegation_id, session_key="agent:main:telegram:dm:12345:678"):
    event = _async_event(delegation_id)
    event["session_key"] = session_key
    event["summary"] = f"Result for {delegation_id}"
    return event


def test_same_tick_async_batch_coalesces_into_one_turn_and_acks_all_rows(
    monkeypatch, isolated_registry,
):
    """Three same-session async completions in one drain -> one synthetic turn.

    All three durable delegation rows must be honestly acknowledged only
    after the single consolidated injection was accepted by the adapter.
    """
    from tools import async_delegation

    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    events = [_distinct_async_event(f"deleg_batch_{i}") for i in range(3)]
    for event in events:
        _persist_pending_completion(event)
        isolated.put(dict(event))

    adapter = _accepting_adapter()
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    assert "3 background subagent delegations" in delivered.text
    for i in range(3):
        assert f"Result for deleg_batch_{i}" in delivered.text
    for event in events:
        row = async_delegation.get_durable_delegation(event["delegation_id"])
        assert row is not None
        assert row["delivery_state"] == "delivered"
    assert isolated.empty()


@pytest.mark.asyncio
async def test_consumed_group_member_cannot_carry_fresh_siblings(
    isolated_registry,
):
    """Consumed rows ACK alone; a fresh row must carry and consume the batch."""
    from tools import async_delegation

    consumed_event = _distinct_async_event("deleg_consumed_primary")
    fresh_events = [
        _distinct_async_event("deleg_fresh_carrier"),
        _distinct_async_event("deleg_fresh_sibling"),
    ]
    for event in (consumed_event, *fresh_events):
        _persist_pending_completion(event)
    assert async_delegation.mark_completion_delivery_consumed(
        "deleg_consumed_primary"
    )

    adapter_events = []

    async def _accept_after_observing_ledger(event):
        adapter_events.append(event)
        assert event.metadata["hermes_completion_delivery_id"] == (
            "async-delegation:deleg_fresh_carrier"
        )

        consumed = async_delegation.get_durable_delegation(
            "deleg_consumed_primary"
        )
        assert consumed is not None and consumed["delivery_state"] == "delivered"
        for delegation_id in ("deleg_fresh_carrier", "deleg_fresh_sibling"):
            fresh = async_delegation.get_durable_delegation(delegation_id)
            assert fresh is not None and fresh["delivery_state"] == "pending"
            assert fresh["consumed_at"] is None

        assert mark_message_consumed(event)
        for delegation_id in ("deleg_fresh_carrier", "deleg_fresh_sibling"):
            consumed_fresh = async_delegation.get_durable_delegation(delegation_id)
            assert consumed_fresh is not None
            assert consumed_fresh["delivery_state"] == "pending"
            assert consumed_fresh["consumed_at"] is not None
        resolve_message_acceptance(event, True)
        return MessageDispatchStatus.ACCEPTED

    adapter = SimpleNamespace(
        handle_message=AsyncMock(side_effect=_accept_after_observing_ledger)
    )
    runner = _runner(adapter)

    assert await runner._deliver_async_delegation_group(
        [consumed_event, *fresh_events]
    ) is True
    adapter.handle_message.assert_awaited_once()
    assert len(adapter_events) == 1

    consumed = async_delegation.get_durable_delegation("deleg_consumed_primary")
    assert consumed is not None and consumed["delivery_state"] == "delivered"
    for delegation_id in ("deleg_fresh_carrier", "deleg_fresh_sibling"):
        fresh = async_delegation.get_durable_delegation(delegation_id)
        assert fresh is not None and fresh["delivery_state"] == "delivered"
        assert fresh["consumed_at"] is not None


def test_coalesced_sibling_ack_rejection_never_emits_false_delivered_state(
    monkeypatch, isolated_registry,
):
    """A sibling ACK CAS failure remains pending and retryable after consumption."""
    from tools import async_delegation

    events = [_distinct_async_event(f"deleg_ack_batch_{i}") for i in range(3)]
    for event in events:
        _persist_pending_completion(event)

    original_complete = async_delegation.complete_event_delivery

    def _reject_one_sibling(evt, claim_id):
        if evt["delegation_id"] == "deleg_ack_batch_1":
            return False
        return original_complete(evt, claim_id)

    monkeypatch.setattr(
        async_delegation,
        "complete_event_delivery",
        _reject_one_sibling,
    )
    adapter = _accepting_adapter()
    runner = _runner(adapter)

    assert asyncio.run(runner._deliver_async_delegation_group(events)) is False

    primary = async_delegation.get_durable_delegation("deleg_ack_batch_0")
    rejected = async_delegation.get_durable_delegation("deleg_ack_batch_1")
    acknowledged = async_delegation.get_durable_delegation("deleg_ack_batch_2")
    assert primary is not None and primary["delivery_state"] == "delivered"
    assert rejected is not None and rejected["delivery_state"] == "pending"
    assert acknowledged is not None and acknowledged["delivery_state"] == "delivered"
    rejected_identity = runner._completion_delivery_identity(events[1])
    assert rejected_identity not in runner._completion_deliveries_delivered
    assert rejected_identity not in runner._completion_deliveries_inflight

    # Restart/replay of the sibling closes only its durable ACK. The consolidated
    # turn already consumed it and must not run a second agent turn.
    monkeypatch.setattr(
        async_delegation,
        "complete_event_delivery",
        original_complete,
    )
    assert asyncio.run(
        runner._deliver_async_delegation_group([events[1]])
    ) is True
    assert adapter.handle_message.await_count == 1
    rejected = async_delegation.get_durable_delegation("deleg_ack_batch_1")
    assert rejected is not None and rejected["delivery_state"] == "delivered"


def test_same_tick_async_events_for_different_sessions_do_not_coalesce(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(_distinct_async_event("deleg_route_a"))
    isolated.put(_distinct_async_event(
        "deleg_route_b", session_key="agent:main:telegram:dm:99999:678",
    ))

    route_delivery_started = asyncio.Event()

    async def _accept_route(event):
        resolve_message_acceptance(event, mark_message_consumed(event))
        route_delivery_started.set()
        return MessageDispatchStatus.ACCEPTED

    adapter = SimpleNamespace(
        handle_message=AsyncMock(side_effect=_accept_route)
    )
    runner = _runner(adapter)
    startup_sleep = True

    async def _stop_after_scheduled_routes_finish(_delay):
        nonlocal startup_sleep
        if startup_sleep:
            startup_sleep = False
            return
        await route_delivery_started.wait()
        await asyncio.gather(*tuple(runner._background_tasks))
        runner._running = False

    monkeypatch.setattr(asyncio, "sleep", _stop_after_scheduled_routes_finish)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    assert adapter.handle_message.await_count == 2
    texts = [call.args[0].text for call in adapter.handle_message.await_args_list]
    assert not any("background subagent delegations" in text for text in texts)
    assert any("deleg_route_a" in text for text in texts)
    assert any("deleg_route_b" in text for text in texts)


@pytest.mark.asyncio
async def test_busy_session_receipt_does_not_block_unrelated_completion(
    monkeypatch, isolated_registry,
):
    """One busy route cannot head-of-line block another route's ACK."""
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(_distinct_async_event("deleg_busy_route"))
    isolated.put(_distinct_async_event(
        "deleg_free_route", session_key="agent:main:telegram:dm:99999:678",
    ))

    busy_entered = asyncio.Event()
    release_busy = asyncio.Event()
    free_accepted = asyncio.Event()

    async def _handle(event):
        if event.source.chat_id == "12345":
            busy_entered.set()
            await release_busy.wait()
        else:
            free_accepted.set()
        resolve_message_acceptance(event, True)
        return MessageDispatchStatus.ACCEPTED

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_handle))
    runner = _runner(adapter)
    real_sleep = asyncio.sleep
    first_sleep = True

    async def _skip_startup_sleep(delay):
        nonlocal first_sleep
        if first_sleep:
            first_sleep = False
            return
        await real_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", _skip_startup_sleep)
    watcher = asyncio.create_task(runner._async_delegation_watcher(interval=0.01))
    try:
        await asyncio.wait_for(busy_entered.wait(), timeout=1)
        await asyncio.wait_for(free_accepted.wait(), timeout=1)
        assert not release_busy.is_set()
    finally:
        runner._running = False
        release_busy.set()
        await asyncio.wait_for(watcher, timeout=1)
        tasks = list(runner._background_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def test_single_async_event_latency_and_text_are_unchanged(
    monkeypatch, isolated_registry,
):
    """A lone completion keeps the plain per-event formatter output."""
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(_distinct_async_event("deleg_single"))

    adapter = _accepting_adapter()
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    assert "background subagent delegations" not in delivered.text
    assert "deleg_single" in delivered.text


def test_failed_coalesced_async_batch_releases_claims_and_retries(
    monkeypatch, isolated_registry,
):
    """A rejected consolidated injection leaves every durable row pending."""
    from tools import async_delegation

    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    events = [_distinct_async_event(f"deleg_retry_{i}") for i in range(2)]
    for event in events:
        _persist_pending_completion(event)
        isolated.put(dict(event))

    adapter = _accepting_adapter(failures=1)
    runner = _runner(adapter)
    sleep_calls = 0
    real_sleep = asyncio.sleep

    async def _stop_after_retry(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if adapter.handle_message.await_count >= 2 and isolated.empty():
            runner._running = False
        elif sleep_calls >= 2000:
            runner._running = False
        await real_sleep(0.001)

    monkeypatch.setattr(asyncio, "sleep", _stop_after_retry)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    # First tick fails as one batch, second tick delivers the same batch.
    assert adapter.handle_message.await_count == 2
    for event in events:
        row = async_delegation.get_durable_delegation(event["delegation_id"])
        assert row is not None
        assert row["delivery_state"] == "delivered"
    assert isolated.empty()


def test_sibling_claimed_by_other_consumer_is_not_double_delivered(
    monkeypatch, isolated_registry,
):
    """A sibling owned elsewhere is excluded from the consolidated turn."""
    from tools import async_delegation

    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    events = [_distinct_async_event(f"deleg_owned_{i}") for i in range(2)]
    for event in events:
        _persist_pending_completion(event)
        isolated.put(dict(event))
    # Simulate another live consumer holding the second row's claim.
    assert async_delegation.claim_completion_delivery(
        events[1]["delegation_id"], "other-consumer:claim",
    )

    adapter = _accepting_adapter()
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    assert "Result for deleg_owned_0" in delivered.text
    assert "Result for deleg_owned_1" not in delivered.text
    row = async_delegation.get_durable_delegation(events[1]["delegation_id"])
    assert row["delivery_state"] == "pending"
