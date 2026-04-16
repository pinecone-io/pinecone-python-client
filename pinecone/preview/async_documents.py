"""Async preview documents data-plane sub-namespace (2026-01.alpha)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgspec

from pinecone._internal.validation import require_in_range, require_non_empty
from pinecone.errors.exceptions import ValidationError
from pinecone.preview._internal.adapters.documents import (
    decode_fetch_response,
    decode_search_response,
)
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.documents import _validate_documents
from pinecone.preview.models.documents import (
    PreviewDocumentFetchResponse,
    PreviewDocumentSearchResponse,
    PreviewDocumentUpsertResponse,
)

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig
    from pinecone._internal.http_client import AsyncHTTPClient
    from pinecone.preview.models.score_by import PreviewScoreByQuery

__all__ = ["AsyncPreviewDocuments"]

_UPSERT_DECODER: msgspec.json.Decoder[PreviewDocumentUpsertResponse] = msgspec.json.Decoder(
    PreviewDocumentUpsertResponse
)


class AsyncPreviewDocuments:
    """Async documents sub-namespace for a preview index.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Access via ``index.documents`` on an
    :class:`~pinecone.preview.async_index.AsyncPreviewIndex`.

    Args:
        http: Async HTTP client from the parent
            :class:`~pinecone.preview.async_index.AsyncPreviewIndex`.
        config: SDK configuration from the parent client.
        host: Data-plane host URL for this index.
    """

    def __init__(self, *, http: AsyncHTTPClient, config: PineconeConfig, host: str) -> None:
        from pinecone._internal.config import PineconeConfig as _PineconeConfig
        from pinecone._internal.http_client import AsyncHTTPClient as _AsyncHTTPClient

        dp_config = _PineconeConfig(
            api_key=config.api_key,
            host=host,
            timeout=config.timeout,
            additional_headers=config.additional_headers,
            source_tag=config.source_tag or "",
            proxy_url=config.proxy_url or "",
            proxy_headers=config.proxy_headers,
            ssl_ca_certs=config.ssl_ca_certs,
            ssl_verify=config.ssl_verify,
            connection_pool_maxsize=config.connection_pool_maxsize,
            retry_config=config.retry_config,
        )
        self._http = _AsyncHTTPClient(dp_config, INDEXES_API_VERSION)

    async def upsert(
        self,
        *,
        namespace: str,
        documents: list[dict[str, Any]],
    ) -> PreviewDocumentUpsertResponse:
        """Upsert documents into a namespace.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Args:
            namespace: Target namespace. Must be a non-empty string.
            documents: 1–100 documents to upsert. Each must contain a non-empty,
                unique ``_id`` string field.

        Returns:
            :class:`~pinecone.preview.models.documents.PreviewDocumentUpsertResponse`
            with ``upserted_count``.

        Raises:
            :exc:`~pinecone.errors.exceptions.ValidationError`: If namespace is empty,
                documents is empty, more than 100 documents, any document is missing
                ``_id``, ``_id`` is not a string, ``_id`` is empty, or ``_id``
                values are not unique within the batch.
        """
        require_non_empty("namespace", namespace)
        _validate_documents(documents)

        response = await self._http.post(
            f"/namespaces/{namespace}/documents/upsert",
            json={"documents": documents},
        )
        return _UPSERT_DECODER.decode(response.content)

    async def search(
        self,
        *,
        namespace: str,
        top_k: int,
        score_by: list[dict[str, Any] | PreviewScoreByQuery],
        include_fields: list[str] | None = None,
        filter: dict[str, Any] | None = None,
    ) -> PreviewDocumentSearchResponse:
        """Search documents in a namespace.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Args:
            namespace: Target namespace. Must be a non-empty string.
            top_k: Number of results to return. Must be between 1 and 10000.
            score_by: Non-empty list of scoring queries. Items may be typed
                :class:`~pinecone.preview.models.score_by.PreviewScoreByQuery`
                structs or plain dicts.
            include_fields: Fields to include in each result. ``None`` returns
                only ``_id`` and ``score``; ``["*"]`` returns all stored fields.
            filter: Optional metadata filter expression.

        Returns:
            :class:`~pinecone.preview.models.documents.PreviewDocumentSearchResponse`
            with ``matches``, ``namespace``, and ``usage``.

        Raises:
            :exc:`~pinecone.errors.exceptions.ValidationError`: If namespace is
                empty, ``top_k`` is outside [1, 10000], or ``score_by`` is empty.
        """
        require_non_empty("namespace", namespace)
        require_in_range("top_k", top_k, 1, 10000)
        require_non_empty("score_by", score_by)

        normalized: list[dict[str, Any]] = [
            msgspec.to_builtins(item) if isinstance(item, msgspec.Struct) else item
            for item in score_by
        ]
        body: dict[str, Any] = {"top_k": top_k, "score_by": normalized}
        if include_fields is not None:
            body["include_fields"] = include_fields
        if filter is not None:
            body["filter"] = filter

        response = await self._http.post(
            f"/namespaces/{namespace}/documents/search",
            json=body,
        )
        return decode_search_response(response.content)

    async def fetch(
        self,
        *,
        namespace: str,
        ids: list[str] | None = None,
        include_fields: list[str] | None = None,
        filter: dict[str, Any] | None = None,
    ) -> PreviewDocumentFetchResponse:
        """Fetch documents from a namespace by ID or filter.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Args:
            namespace: Target namespace. Must be a non-empty string.
            ids: Optional list of document IDs to fetch.
            include_fields: Fields to include in each result. ``None`` returns
                only ``_id``; ``["*"]`` returns all stored fields.
            filter: Optional metadata filter expression.

        Returns:
            :class:`~pinecone.preview.models.documents.PreviewDocumentFetchResponse`
            with ``documents``, ``namespace``, and ``usage``. IDs not present
            in the namespace are silently omitted from ``documents``.

        Raises:
            :exc:`~pinecone.errors.exceptions.ValidationError`: If namespace is empty.
        """
        require_non_empty("namespace", namespace)

        body: dict[str, Any] = {}
        if ids is not None:
            body["ids"] = ids
        if include_fields is not None:
            body["include_fields"] = include_fields
        if filter is not None:
            body["filter"] = filter

        response = await self._http.post(
            f"/namespaces/{namespace}/documents/fetch",
            json=body,
        )
        return decode_fetch_response(response.content)

    async def delete(
        self,
        *,
        namespace: str,
        ids: list[str] | None = None,
        delete_all: bool = False,
        filter: dict[str, Any] | None = None,
    ) -> None:
        """Delete documents from a namespace.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Args:
            namespace: Target namespace. Must be a non-empty string.
            ids: Optional list of document IDs to delete. Mutually exclusive
                with ``delete_all`` and ``filter``.
            delete_all: If ``True``, delete all documents in the namespace.
            filter: Optional metadata filter — delete all matching documents.
                Mutually exclusive with ``ids``.

        Returns:
            ``None`` (server responds with 202 Accepted, empty body).

        Raises:
            :exc:`~pinecone.errors.exceptions.ValidationError`: If namespace is
                empty, none of ``ids``, ``delete_all=True``, or ``filter`` is
                provided, ``ids`` and ``delete_all`` are both provided, or
                ``ids`` and ``filter`` are both provided.
        """
        require_non_empty("namespace", namespace)
        if ids is None and not delete_all and filter is None:
            raise ValidationError(
                "at least one of ids, delete_all=True, or filter must be provided"
            )
        if ids is not None and delete_all:
            raise ValidationError("ids and delete_all are mutually exclusive")
        if ids is not None and filter is not None:
            raise ValidationError("ids and filter are mutually exclusive")

        body: dict[str, Any] = {}
        if ids is not None:
            body["ids"] = ids
        if delete_all:
            body["delete_all"] = True
        if filter is not None:
            body["filter"] = filter

        await self._http.post(
            f"/namespaces/{namespace}/documents/delete",
            json=body,
        )

    def __repr__(self) -> str:
        return "AsyncPreviewDocuments()"
