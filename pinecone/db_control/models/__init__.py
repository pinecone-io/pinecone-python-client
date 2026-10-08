"""Backwards-compatibility shim for :mod:`pinecone.db_control.models`.

Re-exports the control-plane models and specs that used to live here.
Preserved to keep pre-rewrite callers working. New code should import them
from the top-level :mod:`pinecone` namespace. ``PodSpecDefinition`` and
``ServerlessSpecDefinition`` also used to live here; ``index.spec.pod`` and
``index.spec.serverless`` now hold a :class:`~pinecone.PodSpecInfo` and a
:class:`~pinecone.ServerlessSpecInfo` in their place.

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
