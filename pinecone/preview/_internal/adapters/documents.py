"""Preview document response adapters (2026-01.alpha API)."""

from __future__ import annotations

import orjson
import msgspec

from pinecone.preview.models.documents import (
    PreviewDocument,
    PreviewDocumentFetchResponse,
    PreviewDocumentSearchResponse,
    PreviewUsage,
)

__all__ = ["decode_fetch_response", "decode_search_response"]


def decode_search_response(raw: bytes) -> PreviewDocumentSearchResponse:
    """Decode raw JSON bytes from the search endpoint into a typed response."""
    data: dict = orjson.loads(raw)  # type: ignore[type-arg]
    matches = [PreviewDocument(m) for m in data.get("matches", [])]
    namespace: str = data.get("namespace", "")
    usage_data = data.get("usage")
    usage = msgspec.convert(usage_data, PreviewUsage) if usage_data is not None else None
    return PreviewDocumentSearchResponse(matches=matches, namespace=namespace, usage=usage)


def decode_fetch_response(raw: bytes) -> PreviewDocumentFetchResponse:
    """Decode raw JSON bytes from the fetch endpoint into a typed response."""
    data: dict = orjson.loads(raw)  # type: ignore[type-arg]
    raw_docs: dict = data.get("documents", {})  # type: ignore[type-arg]
    documents = {doc_id: PreviewDocument(doc) for doc_id, doc in raw_docs.items()}
    namespace: str = data.get("namespace", "")
    usage_data = data.get("usage")
    usage = msgspec.convert(usage_data, PreviewUsage) if usage_data is not None else None
    return PreviewDocumentFetchResponse(documents=documents, namespace=namespace, usage=usage)
