"""Backwards-compatibility shim for :mod:`pinecone`.

Re-exports public control-plane symbols that used to live under
:mod:`pinecone.control` (which itself was a shim for
:mod:`pinecone.db_control`) before the rewrite. Preserved to keep
pre-rewrite callers working. New code should import from the
canonical top-level :mod:`pinecone` namespace.

:meta private:
"""

from __future__ import annotations

from pinecone.inference.models.index_embed import IndexEmbed
from pinecone.models.backups.list import BackupList, RestoreJobList
from pinecone.models.backups.model import BackupModel, RestoreJobModel
from pinecone.models.collections.description import CollectionDescription
from pinecone.models.collections.list import CollectionList
from pinecone.models.enums import (
    AwsRegion,
    AzureRegion,
    CloudProvider,
    DeletionProtection,
    GcpRegion,
    Metric,
    PodIndexEnvironment,
    PodType,
    VectorType,
)
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.list import IndexList
from pinecone.models.indexes.specs import ByocSpec, PodSpec, ServerlessSpec

__all__ = [
    "AwsRegion",
    "AzureRegion",
    "BackupList",
    "BackupModel",
    "ByocSpec",
    "CloudProvider",
    "CollectionDescription",
    "CollectionList",
    "DeletionProtection",
    "GcpRegion",
    "IndexEmbed",
    "IndexList",
    "IndexModel",
    "Metric",
    "PodIndexEnvironment",
    "PodSpec",
    "PodType",
    "RestoreJobList",
    "RestoreJobModel",
    "ServerlessSpec",
    "VectorType",
]
