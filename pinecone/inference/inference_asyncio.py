"""Backwards-compatibility shim for :mod:`pinecone.async_client.inference`.

Re-exports classes that used to live at :mod:`pinecone.inference.inference_asyncio`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.async_client.inference import AsyncInference as AsyncioInference

__all__ = ["AsyncioInference"]
