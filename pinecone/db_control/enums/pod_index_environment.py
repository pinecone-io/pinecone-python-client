"""Backwards-compatibility shim for :mod:`pinecone.models.enums`.

Re-exports classes that used to live at :mod:`pinecone.db_control.enums.pod_index_environment`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.models.enums import PodIndexEnvironment

__all__ = ["PodIndexEnvironment"]
