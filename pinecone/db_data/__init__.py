"""Backwards-compatibility shim for :mod:`pinecone.index` and :mod:`pinecone.async_client`.

Re-exports classes that used to live at :mod:`pinecone.db_data` before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.async_client.async_index import AsyncIndex as _IndexAsyncio
from pinecone.index import Index as _Index

__all__ = ["_Index", "_IndexAsyncio"]
