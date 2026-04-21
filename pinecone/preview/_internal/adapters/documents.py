"""Preview document response adapters (2026-01.alpha API)."""

from __future__ import annotations

from typing import Any

import httpx
from msgspec import Struct

from pinecone._internal.adapters._decode import decode_response
from pinecone._internal.adapters.vectors_adapter import extract_response_info
from pinecone.models.response_info import ResponseInfo
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
    def to_search_response(response: httpx.Response) -> PreviewDocumentSearchResponse:
        envelope = decode_response(response.content, _SearchEnvelope)
        matches = [PreviewDocument(m) for m in envelope.matches]
        response_info: ResponseInfo = extract_response_info(response)
        return PreviewDocumentSearchResponse(
            matches=matches,
            namespace=envelope.namespace,
            usage=envelope.usage,
            response_info=response_info,
        )

    @staticmethod
    def to_fetch_response(response: httpx.Response) -> PreviewDocumentFetchResponse:
        envelope = decode_response(response.content, _FetchEnvelope)
        documents = {doc_id: PreviewDocument(doc) for doc_id, doc in envelope.documents.items()}
        response_info: ResponseInfo = extract_response_info(response)
        return PreviewDocumentFetchResponse(
            documents=documents,
            namespace=envelope.namespace,
            usage=envelope.usage,
            response_info=response_info,
        )
