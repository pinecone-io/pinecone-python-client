"""Backwards-compatibility shim for :mod:`pinecone.config.config`.

Re-exports classes that used to live at :mod:`pinecone.config.config` before
the ``python-sdk2`` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

# XXX: no canonical equivalent; this shim is the only definition
from __future__ import annotations

from typing import NamedTuple

__all__ = ["Config"]

_EMPTY_HEADERS: dict[str, str] = {}


class Config(NamedTuple):
    """Legacy SDK configuration named tuple.

    Fields mirror the legacy :class:`pinecone.config.config.Config` from
    before the ``python-sdk2`` rewrite.
    """

    api_key: str = ""
    host: str = ""
    proxy_url: str | None = None
    proxy_headers: dict[str, str] | None = None
    ssl_ca_certs: str | None = None
    ssl_verify: bool | None = None
    additional_headers: dict[str, str] = _EMPTY_HEADERS
    source_tag: str | None = None
