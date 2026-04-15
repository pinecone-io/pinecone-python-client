"""Backwards-compatibility shim for :mod:`pinecone.models.collections.description`.

Re-exports classes that used to live at :mod:`pinecone.db_control.models.collection_description`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.models.collections.description import CollectionDescription

__all__ = ["CollectionDescription"]
