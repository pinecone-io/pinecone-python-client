"""Backwards-compatibility shim for :mod:`pinecone.db_control`.

Re-exports classes that used to live at :mod:`pinecone.db_control` before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

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
