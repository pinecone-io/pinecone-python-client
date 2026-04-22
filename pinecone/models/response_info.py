"""Shared HTTP response metadata carrier."""

from __future__ import annotations

from msgspec import Struct, field

from pinecone.models._mixin import StructDictMixin

__all__ = ["ResponseInfo"]


class ResponseInfo(StructDictMixin, Struct, kw_only=True):
    """HTTP response metadata carrier.

    Stores every HTTP response header returned by the server (keys
    lowercased) plus typed convenience properties for the headers the
    SDK promotes to first-class fields.

    Attributes:
        raw_headers (dict[str, str]): All HTTP response headers, keys
            normalized to lowercase. Defaults to an empty dict. Use this
            to read any header the server returns, including headers not
            surfaced by the typed properties below. Prefer the typed
            properties when available — wire header names may change,
            but property semantics are stable.

    Properties:
        request_id (str | None): Server-assigned request identifier read
            from ``x-pinecone-request-id``, or ``None`` if not present.
        lsn_reconciled (int | None): Log sequence number indicating how
            far the index has reconciled, parsed from
            ``x-pinecone-lsn-reconciled``. ``None`` when absent or when
            the header value is not a valid integer.
        lsn_committed (int | None): Log sequence number of the last
            committed write, parsed from ``x-pinecone-lsn-committed``.
            ``None`` when absent or non-integer.
    """

    raw_headers: dict[str, str] = field(default_factory=dict)

    @property
    def request_id(self) -> str | None:
        return self.raw_headers.get("x-pinecone-request-id")

    @property
    def lsn_reconciled(self) -> int | None:
        return _parse_int(self.raw_headers.get("x-pinecone-lsn-reconciled"))

    @property
    def lsn_committed(self) -> int | None:
        return _parse_int(self.raw_headers.get("x-pinecone-lsn-committed"))

    def is_reconciled(self, target: int) -> bool:
        """Return True when the reconciled LSN meets or exceeds *target*."""
        lsn = self.lsn_reconciled
        return lsn is not None and lsn >= target


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
