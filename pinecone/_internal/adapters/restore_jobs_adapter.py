"""Adapter for RestoreJobs API responses."""

from __future__ import annotations

import msgspec
from msgspec import Struct

from pinecone.models.backups.list import RestoreJobList
from pinecone.models.backups.model import RestoreJobModel
from pinecone.models.vectors.responses import Pagination


class _RestoreJobListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-restore-jobs response."""

    data: list[RestoreJobModel] = []
    pagination: Pagination | None = None


class RestoreJobsAdapter:
    """Transforms raw API JSON into RestoreJobModel / RestoreJobList instances."""

    @staticmethod
    def to_restore_job(data: bytes) -> RestoreJobModel:
        """Decode raw JSON bytes into a RestoreJobModel."""
        return msgspec.json.decode(data, type=RestoreJobModel)

    @staticmethod
    def to_restore_job_list(data: bytes) -> RestoreJobList:
        """Decode raw JSON bytes from a list-restore-jobs response into a RestoreJobList."""
        envelope = msgspec.json.decode(data, type=_RestoreJobListEnvelope)
        return RestoreJobList(envelope.data, pagination=envelope.pagination)
