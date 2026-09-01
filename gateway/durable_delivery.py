"""Validated wire identity for durable internal agent turns."""

from __future__ import annotations

import re
from collections.abc import Iterable

DURABLE_DELIVERY_HEADER = "X-Hermes-Durable-Delivery-Id"
DURABLE_DELIVERY_FIELD = "hermes_durable_delivery_ids"
MAX_DURABLE_DELIVERY_IDS = 32
MAX_DURABLE_DELIVERY_ID_LENGTH = 128

UNFINISHED_DURABLE_NONE = "none"
UNFINISHED_DURABLE_OPEN_TAIL = "open_tail"
UNFINISHED_DURABLE_CONFLICT = "conflict"

_DURABLE_DELIVERY_ID_RE = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9._:-]{{0,{MAX_DURABLE_DELIVERY_ID_LENGTH - 1}}}"
)


def validate_origin_profile(value: object) -> str:
    """Return one canonical bounded profile name or raise ``ValueError``."""
    if not isinstance(value, str) or not value:
        raise ValueError("durable origin profile must be a non-empty string")
    from hermes_cli.profiles import validate_profile_name

    validate_profile_name(value)
    return value


def validate_durable_delivery_ids(values: object) -> list[str]:
    """Return a bounded, duplicate-free identity list or raise ``ValueError``."""
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError("durable delivery ids must be a list of strings")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not _DURABLE_DELIVERY_ID_RE.fullmatch(value):
            raise ValueError("invalid durable delivery identity")
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) > MAX_DURABLE_DELIVERY_IDS:
            raise ValueError("too many durable delivery identities")
    return result


def durable_delivery_ids_for_event(event: dict) -> list[str]:
    """Build stable async-delegation wire identities before route selection."""
    if event.get("type") != "async_delegation":
        return []
    coalesced = event.get("_coalesced_delegation_ids") or []
    raw_ids = coalesced or [event.get("delegation_id")]
    values = [f"async-delegation:{value}" for value in raw_ids if value]
    return validate_durable_delivery_ids(values)


def find_durable_admission(
    history: list[dict], delivery_ids: list[str]
) -> tuple[int | None, dict | None]:
    """Locate this delivery's admitted user turn and any terminal assistant."""
    wanted = set(delivery_ids)
    admission_index = None
    for index, item in enumerate(history):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        metadata = item.get("display_metadata") or {}
        carried = set(metadata.get("hermes_completion_delivery_ids") or [])
        if item.get("message_id"):
            carried.add(str(item["message_id"]))
        if wanted & carried:
            admission_index = index
    if admission_index is None:
        return None, None
    turn_end = next(
        (
            index
            for index in range(admission_index + 1, len(history))
            if isinstance(history[index], dict)
            and history[index].get("role") == "user"
        ),
        len(history),
    )
    for item in history[admission_index + 1 : turn_end]:
        if (
            not isinstance(item, dict)
            or item.get("role") != "assistant"
            or item.get("tool_calls")
        ):
            continue
        metadata = item.get("display_metadata") or {}
        carried = metadata.get("hermes_completion_delivery_ids") or []
        if isinstance(carried, list) and wanted.intersection(carried):
            return admission_index, item
    later = [
        item
        for item in history[admission_index + 1 : turn_end]
        if isinstance(item, dict) and item.get("role") in {"assistant", "tool"}
    ]
    terminal = (
        later[-1]
        if later
        and later[-1].get("role") == "assistant"
        and not later[-1].get("tool_calls")
        else None
    )
    return admission_index, terminal


def classify_unfinished_durable_turn(
    history: list[dict],
) -> tuple[str, list[str], int | None]:
    """Classify append-only transcript safety for an unfinished durable turn.

    Only an unfinished durable admission in the final user-turn segment can be
    resumed.  Once another user row exists, appending the old turn's assistant
    would create an ordering that SQLite cannot later reproduce.  Identity-
    linked and legacy terminal assistants are accepted only inside their own
    user-turn boundary.
    """
    open_tail: tuple[list[str], int] | None = None
    for admission_index, item in enumerate(history):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        metadata = item.get("display_metadata") or {}
        raw_ids = metadata.get("hermes_completion_delivery_ids") or []
        if not isinstance(raw_ids, list):
            continue
        delivery_ids = [value for value in raw_ids if isinstance(value, str)]
        if not delivery_ids:
            continue

        wanted = set(delivery_ids)
        turn_end = next(
            (
                index
                for index in range(admission_index + 1, len(history))
                if isinstance(history[index], dict)
                and history[index].get("role") == "user"
            ),
            len(history),
        )
        linked_terminal = False
        for later in history[admission_index + 1 : turn_end]:
            if (
                not isinstance(later, dict)
                or later.get("role") != "assistant"
                or later.get("tool_calls")
            ):
                continue
            linked_ids = (later.get("display_metadata") or {}).get(
                "hermes_completion_delivery_ids"
            ) or []
            if isinstance(linked_ids, list) and wanted.intersection(linked_ids):
                linked_terminal = True
                break
        if linked_terminal:
            continue

        later_in_turn = [
            later
            for later in history[admission_index + 1 : turn_end]
            if isinstance(later, dict)
            and later.get("role") in {"assistant", "tool"}
        ]
        legacy_terminal = bool(
            later_in_turn
            and later_in_turn[-1].get("role") == "assistant"
            and not later_in_turn[-1].get("tool_calls")
        )
        if legacy_terminal:
            continue
        if turn_end < len(history):
            return UNFINISHED_DURABLE_CONFLICT, delivery_ids, admission_index
        open_tail = (delivery_ids, admission_index)

    if open_tail is not None:
        delivery_ids, admission_index = open_tail
        return UNFINISHED_DURABLE_OPEN_TAIL, delivery_ids, admission_index
    return UNFINISHED_DURABLE_NONE, [], None
