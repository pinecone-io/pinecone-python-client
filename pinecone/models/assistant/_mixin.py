"""Backwards-compatibility re-export of :mod:`pinecone.models._mixin`.

The :class:`StructDictMixin` class lives at :mod:`pinecone.models._mixin`.
This module re-exports it (and its legacy ``DictMixin`` alias) so that
assistant model files that import it from this location continue to work.
"""

from __future__ import annotations

from pinecone.models._mixin import (
    DictLikeStruct,
    DictMixin,
    StructDictMixin,
    _struct_to_dict_recursive,
)

__all__ = ["DictLikeStruct", "DictMixin", "StructDictMixin", "_struct_to_dict_recursive"]
