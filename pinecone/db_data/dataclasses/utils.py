"""Backwards-compatibility shim for :mod:`pinecone.db_data.dataclasses.utils`.

Re-exports ``DictLike`` (renamed to ``DictLikeStruct`` in the python-sdk2 rewrite).
Preserved to keep pre-rewrite callers working. New code should import from
:mod:`pinecone.models._mixin`.

:meta private:
"""

from __future__ import annotations

from pinecone.models._mixin import DictLikeStruct as DictLike

__all__ = ["DictLike"]
