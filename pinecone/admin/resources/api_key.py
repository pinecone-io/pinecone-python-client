"""Backwards-compatibility shim for :mod:`pinecone.admin.api_keys`.

Re-exports :class:`pinecone.admin.api_keys.ApiKeys` as
``ApiKeyResource`` for pre-rewrite callers. Preserved to keep
legacy ``from pinecone.admin.resources.api_key import ApiKeyResource``
imports working. New code should use
``from pinecone.admin.api_keys import ApiKeys``.

:meta private:
"""
from __future__ import annotations

from pinecone.admin.api_keys import ApiKeys as ApiKeyResource

__all__ = ["ApiKeyResource"]
