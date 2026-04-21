"""Shared HTTP response metadata carrier."""

from __future__ import annotations

from msgspec import Struct

__all__ = ["ResponseInfo"]


class ResponseInfo(Struct, kw_only=True, gc=False):
    """HTTP response metadata carrier.

    Attributes:
        request_id (str | None): Server-assigned request identifier, or ``None`` if not
            present in the response.
        lsn_reconciled (int | None): Log sequence number indicating how far the index has
            reconciled, or ``None`` if not present in the response headers.
        lsn_committed (int | None): Log sequence number of the last committed write, or
            ``None`` if not present in the response headers.
    """

    request_id: str | None = None
    lsn_reconciled: int | None = None
    lsn_committed: int | None = None

    def is_reconciled(self, target: int) -> bool:
        """Return True when the reconciled LSN meets or exceeds *target*."""
        return self.lsn_reconciled is not None and self.lsn_reconciled >= target
