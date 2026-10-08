"""Backwards-compatibility shim for :mod:`pinecone.config`.

Re-exports the two legacy configuration classes that used to live here:
:class:`~pinecone.config.config.Config`, a named tuple of connection
settings, and :class:`~pinecone.config.pinecone_config.PineconeConfig`, the
static builder that returns one. Both are defined in this package and kept
for pre-rewrite callers. ``ConfigBuilder`` and ``OpenApiConfiguration`` also
used to live here and have no replacement.

New code configures the client through :class:`pinecone.Pinecone` keyword
arguments, or through the unrelated :class:`pinecone.PineconeConfig`
dataclass that the top-level namespace exports.

:meta private:
"""

from __future__ import annotations

from pinecone.config.config import Config
from pinecone.config.pinecone_config import PineconeConfig

__all__ = ["Config", "PineconeConfig"]
