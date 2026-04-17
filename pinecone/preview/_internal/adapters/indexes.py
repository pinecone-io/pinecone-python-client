"""Preview index adapters: create, configure, describe, list (2026-01.alpha API)."""

from __future__ import annotations

from typing import Any

import msgspec
import orjson
from msgspec import Struct

from pinecone._internal.adapters._decode import decode_response
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


class _IndexListEnvelope(Struct, kw_only=True):
    indexes: list[PreviewIndexModel] = []


class PreviewCreateIndexAdapter:
    """Adapter for create_index operation (2026-01.alpha)."""

    @staticmethod
    def to_request(request: PreviewCreateIndexRequest) -> bytes:
        return orjson.dumps(_filter_none(msgspec.to_builtins(request)))

    @staticmethod
    def from_response(data: bytes) -> PreviewIndexModel:
        return decode_response(data, PreviewIndexModel)


class PreviewConfigureIndexAdapter:
    """Adapter for configure_index operation (2026-01.alpha)."""

    @staticmethod
    def to_request(request: PreviewConfigureIndexRequest) -> bytes:
        return orjson.dumps(_filter_none(msgspec.to_builtins(request)))

    @staticmethod
    def from_response(data: bytes) -> PreviewIndexModel:
        return decode_response(data, PreviewIndexModel)


class PreviewDescribeIndexAdapter:
    """Adapter for describe_index operation (2026-01.alpha)."""

    @staticmethod
    def from_response(data: bytes) -> PreviewIndexModel:
        return decode_response(data, PreviewIndexModel)


class PreviewListIndexesAdapter:
    """Adapter for list_indexes operation (2026-01.alpha)."""

    @staticmethod
    def from_response(data: bytes) -> list[PreviewIndexModel]:
        envelope = decode_response(data, _IndexListEnvelope)
        return envelope.indexes


create_adapter = PreviewCreateIndexAdapter()
configure_adapter = PreviewConfigureIndexAdapter()
describe_adapter = PreviewDescribeIndexAdapter()
list_adapter = PreviewListIndexesAdapter()
