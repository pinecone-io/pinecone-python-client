"""Preview index adapters: create, configure, describe, list (2026-01.alpha API)."""

from __future__ import annotations

from typing import Any

import msgspec
import orjson

from pinecone.preview.models.indexes import PreviewIndexModel
from pinecone.preview.models.requests import PreviewConfigureIndexRequest, PreviewCreateIndexRequest

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


def _filter_none(obj: Any) -> Any:
    """Recursively drop None values from dicts. Optional parameters are omitted when None."""
    if isinstance(obj, dict):
        return {k: _filter_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_filter_none(item) for item in obj]
    return obj


class PreviewCreateIndexAdapter:
    """Adapter for create_index operation (2026-01.alpha)."""

    def to_request(self, request: PreviewCreateIndexRequest) -> bytes:
        return orjson.dumps(_filter_none(msgspec.to_builtins(request)))

    def from_response(self, data: dict[str, Any]) -> PreviewIndexModel:
        return msgspec.convert(data, PreviewIndexModel)


class PreviewConfigureIndexAdapter:
    """Adapter for configure_index operation (2026-01.alpha)."""

    def to_request(self, request: PreviewConfigureIndexRequest) -> bytes:
        return orjson.dumps(_filter_none(msgspec.to_builtins(request)))

    def from_response(self, data: dict[str, Any]) -> PreviewIndexModel:
        return msgspec.convert(data, PreviewIndexModel)


class PreviewDescribeIndexAdapter:
    """Adapter for describe_index operation (2026-01.alpha)."""

    def from_response(self, data: dict[str, Any]) -> PreviewIndexModel:
        return msgspec.convert(data, PreviewIndexModel)


class PreviewListIndexesAdapter:
    """Adapter for list_indexes operation (2026-01.alpha)."""

    def from_response(self, data: dict[str, Any]) -> list[PreviewIndexModel]:
        return [msgspec.convert(item, PreviewIndexModel) for item in data["indexes"]]


create_adapter = PreviewCreateIndexAdapter()
configure_adapter = PreviewConfigureIndexAdapter()
describe_adapter = PreviewDescribeIndexAdapter()
list_adapter = PreviewListIndexesAdapter()
