"""Backwards-compatibility shim for :mod:`pinecone.async_client.pinecone`.

Re-exports the :class:`PineconeAsyncio` (alias of
:class:`AsyncPinecone`) client that used to live at
:mod:`pinecone.pinecone_asyncio` before the rewrite. Preserved to
keep pre-rewrite callers working. New code should import
:class:`AsyncPinecone` from :mod:`pinecone` directly.

:meta private:
"""

from __future__ import annotations

from pinecone.async_client.pinecone import AsyncPinecone

PineconeAsyncio = AsyncPinecone

__all__ = ["AsyncPinecone", "PineconeAsyncio"]
