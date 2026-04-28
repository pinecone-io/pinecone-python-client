"""Preview index adapters: create, configure, describe, list (2026-01.alpha API)."""

from __future__ import annotations

import logging
from typing import Any

import msgspec
import orjson
from msgspec import Struct

from pinecone._internal.adapters._decode import decode_response
from pinecone.errors.exceptions import ResponseParsingError
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

_logger = logging.getLogger(__name__)


def _filter_none(obj: Any) -> Any:
    """Recursively drop None values from dicts. Optional parameters are omitted when None."""
    if isinstance(obj, dict):
        return {k: _filter_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_filter_none(item) for item in obj]
    return obj


def _normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Add filterable:true to bare string fields the API rejects without a capability."""
    fields = schema.get("fields")
    if not isinstance(fields, dict):
        return schema
    normalized: dict[str, Any] = {}
    for name, field in fields.items():
        if (
            isinstance(field, dict)
            and field.get("type") == "string"
            and "full_text_search" not in field
            and not field.get("filterable", False)
        ):
            field = {**field, "filterable": True}
        normalized[name] = field
    return {**schema, "fields": normalized}


class _IndexListEnvelope(Struct, kw_only=True):
    indexes: list[PreviewIndexModel] = []


class _RawIndexEnvelope(Struct, kw_only=True):
    # Captures each index as raw JSON bytes so the outer parse always succeeds
    # even when individual items have schema fields with unknown/missing types.
    indexes: list[msgspec.Raw] = []


class PreviewCreateIndexAdapter:
    """Adapter for create_index operation (2026-01.alpha)."""

    @staticmethod
    def to_request(request: PreviewCreateIndexRequest) -> bytes:
        body = _filter_none(msgspec.to_builtins(request))
        if isinstance(body.get("schema"), dict):
            body["schema"] = _normalize_schema(body["schema"])
        return orjson.dumps(body)

    @staticmethod
    def from_response(data: bytes) -> PreviewIndexModel:
        return decode_response(data, PreviewIndexModel)


class PreviewConfigureIndexAdapter:
    """Adapter for configure_index operation (2026-01.alpha)."""

    @staticmethod
    def to_request(request: PreviewConfigureIndexRequest) -> bytes:
        body = _filter_none(msgspec.to_builtins(request))
        if isinstance(body.get("schema"), dict):
            body["schema"] = _normalize_schema(body["schema"])
        return orjson.dumps(body)

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
        # Fast path: strict parse (handles well-formed responses in one step).
        try:
            envelope = decode_response(data, _IndexListEnvelope)
            return envelope.indexes
        except ResponseParsingError:
            pass

        # Resilient path: parse each index independently so a single malformed
        # schema (e.g. a field missing the 'type' discriminator) does not fail
        # the entire list call.  Malformed indexes are skipped with a warning.
        raw_envelope = decode_response(data, _RawIndexEnvelope)
        result: list[PreviewIndexModel] = []
        for raw in raw_envelope.indexes:
            raw_bytes = bytes(raw)
            try:
                result.append(msgspec.json.decode(raw_bytes, type=PreviewIndexModel))
            except (msgspec.ValidationError, msgspec.DecodeError) as exc:
                try:
                    name: str = msgspec.json.decode(raw_bytes, type=dict).get("name", "<unknown>")
                except Exception:
                    name = "<unknown>"
                _logger.warning(
                    "Skipping index %r: cannot parse schema (%s). "
                    "This usually means the index was created with an older or "
                    "experimental API that uses a field type not recognised by "
                    "this SDK version.",
                    name,
                    exc,
                )
        return result


create_adapter = PreviewCreateIndexAdapter()
configure_adapter = PreviewConfigureIndexAdapter()
describe_adapter = PreviewDescribeIndexAdapter()
list_adapter = PreviewListIndexesAdapter()
