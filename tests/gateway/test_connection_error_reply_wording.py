"""Connection failures must be described accurately on chat surfaces.

A mid-response `[Errno 104] Connection reset by peer` is NOT evidence that the
configured endpoint is down — the same turn had already completed an earlier call
against it. Telling the user "the endpoint is not running" sends them debugging a
server that is running (reported 2026-09-06).

The refusal case (nothing listening: ECONNREFUSED / WinError 10061 / no route) IS a
"your endpoint is not up" diagnosis and must keep it, as must the auth, policy and
rate-limit classifications.
"""

import pytest

from gateway.config import Platform
from gateway.run import (
    _gateway_provider_error_reply,
    _sanitize_gateway_final_response,
)

CHAT_PLATFORMS = [Platform.TELEGRAM, "slack", "feishu"]

# Envelopes the agent's terminal path actually produces for a mid-transfer drop: an
# established connection died, which says nothing about whether the endpoint is up.
INTERRUPTED_ENVELOPES = [
    "API call failed after 3 retries: httpx.ReadError: [Errno 104] Connection reset by peer",
    "API call failed after 3 retries: ConnectionResetError: [Errno 104] Connection reset by peer",
    "API call failed after 3 retries: httpx.RemoteProtocolError: peer closed connection "
    "without sending complete message body",
    "API call failed after 3 retries: httpx.ReadError: server disconnected without sending a response",
]

# Envelopes that really do mean "nothing is listening at the configured endpoint".
UNREACHABLE_ENVELOPES = [
    "API call failed after 3 retries: httpx.ConnectError: [Errno 111] Connection refused",
    "API call failed after 3 retries: ConnectionError: [WinError 10061] No connection could "
    "be made because the target machine actively refused it",
    "API call failed after 3 retries: httpx.ConnectError: [Errno 113] No route to host",
]

# Connection-shaped, but the SDK flattened the cause away — neither diagnosis is supported.
AMBIGUOUS_ENVELOPES = [
    "API call failed after 3 retries: openai.APIConnectionError: Connection error.",
    "❌ API failed after 3 retries — openai.APIConnectionError: Connection error.",
]

# Wording that ASSERTS the endpoint is down / not started.
_DOWN_DIAGNOSIS_CLAIMS = ("is not running", "not responding", "is unreachable", "not started")


def _claims_endpoint_is_down(reply: str) -> bool:
    return any(claim in reply.lower() for claim in _DOWN_DIAGNOSIS_CLAIMS)


@pytest.mark.parametrize("envelope", INTERRUPTED_ENVELOPES)
def test_interrupted_connection_is_not_diagnosed_as_a_dead_endpoint(envelope):
    """A reset/dropped response must be reported as an interruption, not a dead server."""
    reply = _gateway_provider_error_reply(envelope)

    assert reply
    assert not _claims_endpoint_is_down(reply), reply


@pytest.mark.parametrize("envelope", UNREACHABLE_ENVELOPES)
def test_refused_connection_keeps_the_endpoint_down_diagnosis(envelope):
    """A refused/unroutable connect is exactly the case the old wording was written for."""
    reply = _gateway_provider_error_reply(envelope)

    assert _claims_endpoint_is_down(reply), reply


@pytest.mark.parametrize("envelope", AMBIGUOUS_ENVELOPES)
def test_ambiguous_connection_error_asserts_neither_cause(envelope):
    """An SDK-flattened ``Connection error.`` supports no diagnosis — so make none."""
    reply = _gateway_provider_error_reply(envelope)

    assert not _claims_endpoint_is_down(reply), reply
    # It still has to be actionable: the user is told what to check, not what is broken.
    assert "reachable" in reply.lower(), reply


def test_the_three_connection_causes_are_distinct_categories():
    """The causes must not collapse onto one message again."""
    interrupted = {_gateway_provider_error_reply(e) for e in INTERRUPTED_ENVELOPES}
    unreachable = {_gateway_provider_error_reply(e) for e in UNREACHABLE_ENVELOPES}
    ambiguous = {_gateway_provider_error_reply(e) for e in AMBIGUOUS_ENVELOPES}

    assert len(interrupted) == len(unreachable) == len(ambiguous) == 1
    assert len(interrupted | unreachable | ambiguous) == 3


@pytest.mark.parametrize(
    "envelope, expected_marker",
    [
        (
            "API call failed after 3 retries: HTTP 401 Unauthorized: incorrect api key provided",
            "authentication",
        ),
        (
            "API call failed after 3 retries: HTTP 429: rate limit exceeded for this model",
            "rate-limiting",
        ),
        (
            "API call failed after 3 retries: HTTP 400: request blocked under the provider "
            "safety policy",
            "rejected",
        ),
    ],
    ids=["auth", "rate_limit", "policy"],
)
def test_other_provider_error_classifications_are_preserved(envelope, expected_marker):
    """Auth beats policy beats rate-limit beats connection — that ordering still holds."""
    reply = _gateway_provider_error_reply(envelope)

    assert expected_marker in reply.lower(), reply


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
def test_reset_final_response_is_sanitized_and_secret_free(platform):
    """The reset envelope still goes through redaction + the safe-category rewrite."""
    raw = (
        "API call failed after 3 retries: httpx.ReadError: [Errno 104] Connection reset "
        "by peer (Authorization: Bearer sk-ABCDEF0123456789abcdef0123)"
    )

    sanitized = _sanitize_gateway_final_response(platform, raw)

    assert "sk-ABCDEF" not in sanitized
    assert "Errno 104" not in sanitized
    assert not _claims_endpoint_is_down(sanitized), sanitized
    assert sanitized.strip()


@pytest.mark.parametrize("platform", ["local", "api_server", "webhook"])
def test_programmatic_surfaces_keep_the_raw_connection_error(platform):
    """CLI/API consumers still need the bottom exception, not a chat-safe category."""
    raw = "API call failed after 3 retries: httpx.ReadError: [Errno 104] Connection reset by peer"

    assert _sanitize_gateway_final_response(platform, raw) == raw
