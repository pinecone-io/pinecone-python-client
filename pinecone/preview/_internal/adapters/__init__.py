"""Preview index adapters (2026-01.alpha API)."""

from pinecone.preview._internal.adapters.indexes import (
    PreviewConfigureIndexAdapter,
    PreviewCreateIndexAdapter,
    PreviewDescribeIndexAdapter,
    PreviewListIndexesAdapter,
    configure_adapter,
    create_adapter,
    describe_adapter,
    list_adapter,
)

__all__ = [
    "PreviewConfigureIndexAdapter",
    "PreviewCreateIndexAdapter",
    "PreviewDescribeIndexAdapter",
    "PreviewListIndexesAdapter",
    "configure_adapter",
    "create_adapter",
    "describe_adapter",
    "list_adapter",
]
