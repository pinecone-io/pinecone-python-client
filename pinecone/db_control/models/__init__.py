"""Backwards-compatibility shim for :mod:`pinecone.db_control.models`.

Re-exports classes that used to live at :mod:`pinecone.db_control.models` before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

__all__: list[str] = []
