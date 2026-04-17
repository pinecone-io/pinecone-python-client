"""Preview document response adapters (2026-01.alpha API)."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone._internal.adapters._decode import decode_response
from pinecone.preview.models.documents import (
    PreviewDocument,
    PreviewDocumentFetchResponse,
    PreviewDocumentSearchResponse,
    PreviewUsage,
)

__all__ = ["PreviewDocumentsAdapter"]


class _SearchEnvelope(Struct, kw_only=True):
    matches: list[Any] = []
    namespace: str = ""
    usage: PreviewUsage | None = None


class _FetchEnvelope(Struct, kw_only=True):
    documents: dict[str, Any] = {}
    namespace: str = ""
    usage: PreviewUsage | None = None


class PreviewDocumentsAdapter:
    """Adapter for preview document search and fetch operations."""

    @staticmethod
    def to_search_response(data: bytes) -> PreviewDocumentSearchResponse:
        envelope = decode_response(data, _SearchEnvelope)
        matches = [PreviewDocument(m) for m in envelope.matches]
        return PreviewDocumentSearchResponse(
            matches=matches, namespace=envelope.namespace, usage=envelope.usage
        )

    @staticmethod
    def to_fetch_response(data: bytes) -> PreviewDocumentFetchResponse:
        envelope = decode_response(data, _FetchEnvelope)
        documents = {doc_id: PreviewDocument(doc) for doc_id, doc in envelope.documents.items()}
        return PreviewDocumentFetchResponse(
            documents=documents, namespace=envelope.namespace, usage=envelope.usage
        )
