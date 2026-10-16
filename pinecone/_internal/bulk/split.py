"""Halve-and-resend for batches the server rejects as too large.

A batcher that slices by count cannot know how many bytes a slice encodes to,
and the server's request-size cap is the server's to state. So this module
does not predict the cap: a 413 on a multi-item batch is answered by sending
each half separately, recursively. A batch that fits costs one request; one
that does not costs a bounded number of extra requests.

Only a REST ``ApiError`` with ``status_code`` 413 triggers a split. Once
*expired* reports the caller's deadline has passed, pieces not yet sent are
recorded as ``unsent`` instead of being sent: the halving must not run past
``total_timeout``.

A 413 on a single item is final — no further split can help — and is recorded
as that item's failure, in the same shape any other rejection has.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pinecone._internal.bulk.classify import DISPOSITION_REJECTED, DISPOSITION_UNSENT

_PAYLOAD_TOO_LARGE = 413

UNSENT_PIECE_GROUP_KEY = "total_timeout expired before this split piece was submitted"


@dataclass
class SplitOutcome:
    """What sending one batch produced: the result of every request that was
    accepted, and one ``(items, exception, disposition)`` triple per piece that was not."""

    results: list[Any] = field(default_factory=list)
    sent_item_count: int = 0
    failures: list[tuple[list[dict[str, Any]], Exception, str]] = field(default_factory=list)


def _unsent(piece: list[dict[str, Any]]) -> Exception:
    from pinecone.errors.exceptions import PineconeTimeoutError

    return PineconeTimeoutError(f"{UNSENT_PIECE_GROUP_KEY}; {len(piece)} items were not sent")


def _is_payload_too_large(exc: Exception, batch: list[dict[str, Any]]) -> bool:
    return len(batch) > 1 and getattr(exc, "status_code", None) == _PAYLOAD_TOO_LARGE


def send_halving(
    operation: Callable[[list[dict[str, Any]]], Any],
    batch: list[dict[str, Any]],
    *,
    split: bool,
    expired: Callable[[], bool] | None = None,
) -> SplitOutcome:
    outcome = SplitOutcome()

    def _send(piece: list[dict[str, Any]], split_part: bool = False) -> None:
        if split_part and expired is not None and expired():
            outcome.failures.append((piece, _unsent(piece), DISPOSITION_UNSENT))
            return
        try:
            result = operation(piece)
        except Exception as exc:
            if split and _is_payload_too_large(exc, piece):
                mid = len(piece) // 2
                _send(piece[:mid], split_part=True)
                _send(piece[mid:], split_part=True)
            else:
                outcome.failures.append((piece, exc, DISPOSITION_REJECTED))
        else:
            outcome.results.append(result)
            outcome.sent_item_count += len(piece)

    _send(batch)
    return outcome


async def send_halving_async(
    operation: Callable[[list[dict[str, Any]]], Awaitable[Any]],
    batch: list[dict[str, Any]],
    *,
    split: bool,
    expired: Callable[[], bool] | None = None,
) -> SplitOutcome:
    outcome = SplitOutcome()

    async def _send(piece: list[dict[str, Any]], split_part: bool = False) -> None:
        if split_part and expired is not None and expired():
            outcome.failures.append((piece, _unsent(piece), DISPOSITION_UNSENT))
            return
        try:
            result = await operation(piece)
        except Exception as exc:
            if split and _is_payload_too_large(exc, piece):
                mid = len(piece) // 2
                await _send(piece[:mid], split_part=True)
                await _send(piece[mid:], split_part=True)
            else:
                outcome.failures.append((piece, exc, DISPOSITION_REJECTED))
        else:
            outcome.results.append(result)
            outcome.sent_item_count += len(piece)

    await _send(batch)
    return outcome
