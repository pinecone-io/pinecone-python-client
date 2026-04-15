"""Backwards-compatibility shim for :mod:`pinecone.models.backups.model`.

Re-exports classes that used to live at :mod:`pinecone.db_control.models.restore_job_model`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.models.backups.model import RestoreJobModel

__all__ = ["RestoreJobModel"]
