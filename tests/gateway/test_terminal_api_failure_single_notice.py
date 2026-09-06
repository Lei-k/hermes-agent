"""A terminal provider failure must produce exactly ONE user-visible chat bubble.

Reported 2026-09-06: three `httpx.ReadError: [Errno 104] Connection reset by peer`
attempts exhausted the retry budget on a Telegram turn and the user saw the *same*
sanitized connection warning twice.

The duplicate is structural, not platform-specific: ``max_retries_exhausted_result``
emits a terminal status through ``status_callback`` AND returns a ``final_response``
carrying the same failure envelope. On chat surfaces the gateway rewrites both through
the same provider-error reply table, so both bubbles render identically.

These tests drive the REAL terminal path (real ``AIAgent``, real classifier, real
``httpx.ReadError``) and pipe its two outputs through the REAL gateway filters, so they
assert the delivery contract rather than any particular wording.
"""

import httpx
import pytest
from unittest.mock import MagicMock, patch

from agent.error_classifier import classify_api_error
from agent.turn_recovery import max_retries_exhausted_result
from gateway.config import Platform
from gateway.run import (
    _prepare_gateway_status_message,
    _sanitize_gateway_final_response,
)
from run_agent import AIAgent

CHAT_PLATFORMS = [Platform.TELEGRAM, "slack", "feishu", "irc"]
PROGRAMMATIC_SURFACES = ["local", "api_server", "webhook"]


def _make_agent():
    """A minimal real AIAgent (no network, no live endpoint probing)."""
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch(
            "agent.context_compressor.get_model_context_length",
            return_value=200_000,
        ),
    ):
        agent = AIAgent(
            api_key="test-key-12345678",
            base_url="https://my-llm.example.com/v1",
            provider="custom",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._persist_session = MagicMock()
    return agent


def _run_terminal_failure(agent, api_error, *, is_rate_limited=False):
    """Run the real retries-exhausted terminal path; return (statuses, final_response)."""
    statuses: list[tuple[str, str]] = []
    agent.status_callback = lambda kind, message: statuses.append((kind, message))
    classified = classify_api_error(api_error)
    result = max_retries_exhausted_result(
        agent,
        api_error,
        classified,
        max_retries=3,
        is_rate_limited=is_rate_limited,
        error_msg=str(api_error).lower(),
        api_kwargs=None,
        api_messages=[{"role": "user", "content": "hi"}],
        messages=[{"role": "user", "content": "hi"}],
        conversation_history=None,
        api_call_count=2,
        approx_tokens=51_987,
        provider="openai-codex",
        base_url="https://example.invalid/backend-api/codex",
        model="test-model",
    )
    return statuses, result["final_response"]


def _delivered_bubbles(platform, statuses, final_response):
    """Everything this platform would actually render for the failed turn."""
    bubbles = [
        prepared
        for kind, message in statuses
        if (prepared := _prepare_gateway_status_message(platform, kind, message))
    ]
    final = _sanitize_gateway_final_response(platform, final_response)
    if final:
        bubbles.append(final)
    return bubbles


def _reset_error():
    """The exact live failure shape: a mid-response read reset."""
    return httpx.ReadError("[Errno 104] Connection reset by peer")


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
def test_connection_reset_terminal_failure_renders_one_bubble(platform):
    """The reported shape: retries exhausted on a connection reset."""
    agent = _make_agent()
    statuses, final_response = _run_terminal_failure(agent, _reset_error())

    # The agent still reports the failure on both channels — that is its contract.
    assert statuses, "terminal failure must still emit a status for local/CLI surfaces"
    assert final_response

    bubbles = _delivered_bubbles(platform, statuses, final_response)

    assert len(bubbles) == 1, f"user saw {len(bubbles)} bubbles: {bubbles}"
    assert "reset by peer" not in bubbles[0].lower()


@pytest.mark.parametrize(
    "api_error",
    [
        httpx.ReadError("[Errno 104] Connection reset by peer"),
        httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
        httpx.ConnectError("[Errno 111] Connection refused"),
        RuntimeError("HTTP 500: internal server error"),
    ],
    ids=["read_reset", "remote_protocol", "connect_refused", "http_500"],
)
def test_every_terminal_failure_shape_renders_one_bubble(api_error):
    """No terminal provider failure envelope may be delivered twice."""
    agent = _make_agent()
    statuses, final_response = _run_terminal_failure(agent, api_error)

    bubbles = _delivered_bubbles(Platform.TELEGRAM, statuses, final_response)

    assert len(bubbles) == 1, f"user saw {len(bubbles)} bubbles: {bubbles}"


def test_rate_limited_terminal_failure_renders_one_bubble():
    """The rate-limit terminal variant must keep the same single-bubble contract."""
    agent = _make_agent()
    error = RuntimeError("HTTP 429: too many requests, please retry after 20s")
    statuses, final_response = _run_terminal_failure(agent, error, is_rate_limited=True)

    bubbles = _delivered_bubbles(Platform.TELEGRAM, statuses, final_response)

    assert len(bubbles) == 1, f"user saw {len(bubbles)} bubbles: {bubbles}"
    assert "rate" in bubbles[0].lower() or "limit" in bubbles[0].lower()


def test_non_retryable_terminal_failure_renders_one_bubble():
    """The sibling terminal path (non-retryable 4xx) has the same two outputs."""
    from agent.turn_recovery import nonretryable_client_error_result

    agent = _make_agent()
    statuses: list[tuple[str, str]] = []
    agent.status_callback = lambda kind, message: statuses.append((kind, message))
    api_error = RuntimeError("HTTP 401: Incorrect API key provided: sk-live_abcdefghij0123456789")
    api_error.status_code = 401

    result = nonretryable_client_error_result(
        agent,
        api_error,
        classify_api_error(api_error),
        status_code=401,
        api_kwargs=None,
        api_messages=[{"role": "user", "content": "hi"}],
        messages=[{"role": "user", "content": "hi"}],
        conversation_history=None,
        api_call_count=1,
        approx_tokens=100,
        provider="openai-codex",
        base_url="https://example.invalid/backend-api/codex",
        model="test-model",
    )

    bubbles = _delivered_bubbles(Platform.TELEGRAM, statuses, result["final_response"])

    assert len(bubbles) == 1, f"user saw {len(bubbles)} bubbles: {bubbles}"
    assert "sk-live_" not in bubbles[0]


@pytest.mark.parametrize("surface", PROGRAMMATIC_SURFACES)
def test_programmatic_surfaces_keep_both_diagnostic_channels(surface):
    """CLI/API/webhook consumers keep the raw status stream AND the raw final."""
    agent = _make_agent()
    statuses, final_response = _run_terminal_failure(agent, _reset_error())

    prepared = [
        _prepare_gateway_status_message(surface, kind, message)
        for kind, message in statuses
    ]

    assert [p for p in prepared if p] == [m for _, m in statuses]
    assert _sanitize_gateway_final_response(surface, final_response) == final_response


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
def test_progress_and_unrelated_warnings_still_reach_chat(platform):
    """Suppressing the duplicate must not silence ordinary work/warning statuses."""
    keepers = [
        ("lifecycle", "still on it"),
        ("lifecycle", "⏳ Working — 3 min"),
        ("warn", "⚠️ The command you approved touches files outside the workspace."),
        ("warn", "🛑 Tool budget exhausted for this turn."),
    ]

    for kind, message in keepers:
        assert _prepare_gateway_status_message(platform, kind, message) == message


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
def test_context_overflow_warning_still_reaches_chat(platform):
    """The context-overflow advisory is a real user warning, not a provider envelope."""
    from agent.conversation_compression import (
        CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE,
    )

    message = CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
        tokens=85_000, threshold=72_000, reason="ineffective"
    )

    assert _prepare_gateway_status_message(platform, "warn", message) == message


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
def test_terminal_failure_status_never_leaks_provider_detail(platform):
    """Whatever the delivery decision, raw provider bodies/secrets never reach chat."""
    raw = (
        "❌ API failed after 3 retries — HTTP 401 Unauthorized: "
        "Authorization: Bearer sk-ABCDEF0123456789abcdef0123 request_id=req_9"
    )

    prepared = _prepare_gateway_status_message(platform, "lifecycle", raw)

    if prepared is not None:
        assert "sk-ABCDEF" not in prepared
        assert "req_9" not in prepared
        assert "HTTP 401" not in prepared
