"""Backwards-compatibility shim for :mod:`pinecone.db_control.enums`.

Re-exports the control-plane enums that used to live here. Preserved to keep
pre-rewrite callers working. New code should import them from the top-level
:mod:`pinecone` namespace.

:meta private:
"""

from __future__ import annotations

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

__all__ = [
    "AwsRegion",
    "AzureRegion",
    "CloudProvider",
    "DeletionProtection",
    "GcpRegion",
    "Metric",
    "PodIndexEnvironment",
    "PodType",
    "VectorType",
]
