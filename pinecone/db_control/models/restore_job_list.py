"""Backwards-compatibility shim for :mod:`pinecone.models.backups.list`.

Re-exports classes that used to live at :mod:`pinecone.db_control.models.restore_job_list`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.models.backups.list import RestoreJobList

__all__ = ["RestoreJobList"]
