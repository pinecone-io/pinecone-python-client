"""Backwards-compatibility shim for :mod:`pinecone.db_control`.

Re-exports the control-plane enums, models, and specs that used to live
here, matching what :mod:`pinecone.control` exports. Preserved to keep
pre-rewrite callers working. New code should import them from the top-level
:mod:`pinecone` namespace. ``DBControl``, ``DBControlAsyncio``,
``PodSpecDefinition``, ``ServerlessSpecDefinition``, ``ConfigureIndexEmbed``,
and ``CreateIndexForModelEmbedTypedDict`` also used to live here and are not
re-exported; :doc:`/migration/v10-migration` covers what replaces each.

:meta private:
"""

from __future__ import annotations

from pinecone.db_control.enums import (
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
from pinecone.db_control.models import (
    BackupList,
    BackupModel,
    ByocSpec,
    CollectionDescription,
    CollectionList,
    IndexEmbed,
    IndexList,
    IndexModel,
    PodSpec,
    RestoreJobList,
    RestoreJobModel,
    ServerlessSpec,
)

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
