"""Backwards-compatibility shim for :mod:`pinecone._internal.config`.

Re-exports classes that used to live at :mod:`pinecone.config` before
the ``python-sdk2`` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

__all__: list[str] = []
