"""Backwards-compatibility shim for :mod:`pinecone.db_control.models`.

Re-exports classes that used to live at :mod:`pinecone.db_control.models` before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.inference.models.index_embed import IndexEmbed
from pinecone.models.backups.list import BackupList, RestoreJobList
from pinecone.models.backups.model import BackupModel, RestoreJobModel
from pinecone.models.collections.description import CollectionDescription
from pinecone.models.collections.list import CollectionList
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.list import IndexList
from pinecone.models.indexes.specs import ByocSpec, PodSpec, ServerlessSpec

__all__ = [
    "BackupList",
    "BackupModel",
    "ByocSpec",
    "CollectionDescription",
    "CollectionList",
    "IndexEmbed",
    "IndexList",
    "IndexModel",
    "PodSpec",
    "RestoreJobList",
    "RestoreJobModel",
    "ServerlessSpec",
]
