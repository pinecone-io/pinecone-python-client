"""Preview documents data-plane sub-namespace (2026-01.alpha)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgspec

from pinecone._internal.adapters.vectors_adapter import extract_response_info
from pinecone._internal.batch import batch_execute
from pinecone._internal.validation import require_in_range, require_non_empty
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.batch import (
    BatchResult,  # SDK utility result, not wire-shape — see preview-channel.md § Type isolation
)
from pinecone.preview._internal.adapters.documents import PreviewDocumentsAdapter
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.models.documents import (
    PreviewDocumentFetchResponse,
    PreviewDocumentSearchResponse,
    PreviewDocumentUpsertResponse,
)

if TYPE_CHECKING:
    from pinecone._internal.config import PineconeConfig
    from pinecone.preview.models.score_by import PreviewScoreByQuery

__all__ = ["PreviewDocuments"]

_UPSERT_DECODER: msgspec.json.Decoder[PreviewDocumentUpsertResponse] = msgspec.json.Decoder(
    PreviewDocumentUpsertResponse
)


def _validate_documents(documents: list[dict[str, Any]]) -> None:
    require_non_empty("documents", documents)
    if len(documents) > 100:
        raise PineconeValueError("documents must contain at most 100 items")
    seen_ids: set[str] = set()
    for i, doc in enumerate(documents):
        if "_id" not in doc:
            raise PineconeValueError(f"document at index {i} is missing required '_id' field")
        doc_id = doc["_id"]
        if not isinstance(doc_id, str):
            raise PineconeValueError(f"document at index {i} has non-string '_id' value")
        if not doc_id:
            raise PineconeValueError(f"document at index {i} has empty '_id' value")
        if doc_id in seen_ids:
            raise PineconeValueError(f"document at index {i} has duplicate '_id': {doc_id!r}")
        seen_ids.add(doc_id)


class PreviewDocuments:
    """Documents sub-namespace for a preview index.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Access via ``index.documents`` on a :class:`~pinecone.preview.index.PreviewIndex`.

    Args:
        config: SDK configuration from the parent client.
        host: Data-plane host URL for this index.

    Examples:
        Get the documents namespace from a preview index and upsert a document:

        >>> from pinecone import Pinecone
        >>> pc = Pinecone(api_key="your-api-key")
        >>> index = pc.preview.index(name="articles-en-preview")
        >>> docs = index.documents
        >>> response = docs.upsert(
        ...     namespace="articles-en",
        ...     documents=[
        ...         {"_id": "article-101", "title": "Introduction to vectors", "category": "tech"},
        ...     ],
        ... )
        >>> response.upserted_count
        1
    """

    def __init__(self, *, config: PineconeConfig, host: str) -> None:
        from pinecone._internal.config import PineconeConfig as _PineconeConfig
        from pinecone._internal.http_client import HTTPClient as _HTTPClient

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
        self._http = _HTTPClient(dp_config, INDEXES_API_VERSION)

    def close(self) -> None:
        """Close the underlying HTTP client. Idempotent.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.
        """
        self._http.close()

    def upsert(
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
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If namespace is empty,
                documents is empty, more than 100 documents, any document is missing
                ``_id``, ``_id`` is not a string, ``_id`` is empty, or ``_id``
                values are not unique within the batch.

        Examples:
            Upsert a single document with text fields:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> index = pc.preview.index(name="articles-en-preview")
            >>> response = index.documents.upsert(
            ...     namespace="articles-en",
            ...     documents=[
            ...         {"_id": "article-101", "title": "Intro to vectors", "category": "tech"},
            ...     ],
            ... )
            >>> response.upserted_count
            1

            Upsert multiple documents with embeddings and metadata:

            >>> response = index.documents.upsert(
            ...     namespace="articles-en",
            ...     documents=[
            ...         {
            ...             "_id": "article-101",
            ...             "title": "Introduction to vectors",
            ...             "embedding": [0.012, -0.087, 0.153],
            ...             "category": "tech",
            ...         },
            ...         {
            ...             "_id": "article-102",
            ...             "title": "Advanced retrieval methods",
            ...             "embedding": [0.045, 0.021, -0.064],
            ...             "category": "research",
            ...         },
            ...     ],
            ... )
            >>> response.upserted_count
            2
        """
        require_non_empty("namespace", namespace)
        _validate_documents(documents)

        response = self._http.post(
            f"/namespaces/{namespace}/documents/upsert",
            json={"documents": documents},
        )
        result = _UPSERT_DECODER.decode(response.content)
        result.response_info = extract_response_info(response)
        return result

    def batch_upsert(
        self,
        *,
        namespace: str,
        documents: list[dict[str, Any]],
        batch_size: int = 100,
        max_workers: int = 4,
        show_progress: bool = True,
    ) -> BatchResult:
        """Upsert a large list of documents in parallel batches.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Args:
            namespace: Target namespace. Must be a non-empty string.
            documents: Documents to upsert. Each must contain a non-empty,
                unique ``_id`` string field.
            batch_size: Maximum documents per request (1–100, default 100).
            max_workers: Thread pool size for concurrent requests (1–64, default 4).
            show_progress: Display a tqdm progress bar when installed.

        Returns:
            :class:`~pinecone.models.batch.BatchResult` with aggregated success
            and failure counts. Per-batch HTTP failures are captured in
            ``result.errors`` rather than raised.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If namespace is
                empty, documents is empty, batch_size is outside [1, 100], or
                max_workers is outside [1, 64].

        Examples:
            Batch upsert a large corpus of documents:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> index = pc.preview.index(name="articles-en-preview")
            >>> documents = [
            ...     {"_id": f"article-{i}", "title": f"Article {i}", "embedding": [0.012, -0.087]}
            ...     for i in range(500)
            ... ]
            >>> result = index.documents.batch_upsert(
            ...     namespace="articles-en",
            ...     documents=documents,
            ...     batch_size=50,
            ...     max_workers=8,
            ... )
            >>> result.success_count
            500
            >>> result.error_count
            0
        """
        require_non_empty("namespace", namespace)
        require_non_empty("documents", documents)
        require_in_range("batch_size", batch_size, 1, 100)
        require_in_range("max_workers", max_workers, 1, 64)

        return batch_execute(
            items=documents,
            operation=lambda chunk: self.upsert(namespace=namespace, documents=chunk),
            batch_size=batch_size,
            max_workers=max_workers,
            show_progress=show_progress,
            desc="Upserting",
        )

    def search(
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
                only ``_id`` and ``_score``; ``["*"]`` returns all stored fields.
            filter: Optional metadata filter expression.

        Returns:
            :class:`~pinecone.preview.models.documents.PreviewDocumentSearchResponse`
            with ``matches``, ``namespace``, and ``usage``.

        Raises:
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If namespace is
                empty, ``top_k`` is outside [1, 10000], or ``score_by`` is empty.

        Examples:
            Search with a dense vector query:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> index = pc.preview.index(name="articles-en-preview")
            >>> results = index.documents.search(
            ...     namespace="articles-en",
            ...     top_k=5,
            ...     score_by=[{"field": "embedding", "query": [0.012, -0.087, 0.153]}],
            ... )
            >>> results.matches[0]._id
            'article-42'

            Search with a metadata filter and select specific fields:

            >>> results = index.documents.search(
            ...     namespace="articles-en",
            ...     top_k=10,
            ...     score_by=[{"field": "embedding", "query": [0.012, -0.087, 0.153]}],
            ...     include_fields=["_id", "title", "category"],
            ...     filter={"category": "tech"},
            ... )
        """
        require_non_empty("namespace", namespace)
        require_in_range("top_k", top_k, 1, 10000)
        require_non_empty("score_by", score_by)

        normalized: list[dict[str, Any]] = [
            msgspec.to_builtins(item) if isinstance(item, msgspec.Struct) else item
            for item in score_by
        ]
        body: dict[str, Any] = {
            "top_k": top_k,
            "score_by": normalized,
        }
        if include_fields is not None:
            body["include_fields"] = include_fields
        if filter is not None:
            body["filter"] = filter

        response = self._http.post(
            f"/namespaces/{namespace}/documents/search",
            json=body,
        )
        return PreviewDocumentsAdapter.to_search_response(response)

    def fetch(
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
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If namespace is empty.

        Examples:
            Fetch specific documents by ID:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> index = pc.preview.index(name="articles-en-preview")
            >>> response = index.documents.fetch(
            ...     namespace="articles-en",
            ...     ids=["article-101", "article-102"],
            ...     include_fields=["_id", "title", "category"],
            ... )
            >>> len(response.documents)
            2

            Fetch all documents matching a filter with all fields:

            >>> response = index.documents.fetch(
            ...     namespace="articles-en",
            ...     include_fields=["*"],
            ...     filter={"category": "tech"},
            ... )
        """
        require_non_empty("namespace", namespace)

        body: dict[str, Any] = {}
        if ids is not None:
            body["ids"] = ids
        if include_fields is not None:
            body["include_fields"] = include_fields
        if filter is not None:
            body["filter"] = filter

        response = self._http.post(
            f"/namespaces/{namespace}/documents/fetch",
            json=body,
        )
        return PreviewDocumentsAdapter.to_fetch_response(response)

    def delete(
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
            :exc:`~pinecone.errors.exceptions.PineconeValueError`: If namespace is
                empty, none of ``ids``, ``delete_all=True``, or ``filter`` is
                provided, ``ids`` and ``delete_all`` are both provided, or
                ``ids`` and ``filter`` are both provided.

        Examples:
            Delete specific documents by ID:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> index = pc.preview.index(name="articles-en-preview")
            >>> index.documents.delete(
            ...     namespace="articles-en",
            ...     ids=["article-101", "article-102"],
            ... )

            Delete all documents matching a filter:

            >>> index.documents.delete(
            ...     namespace="articles-en",
            ...     filter={"category": "draft"},
            ... )

            Delete all documents in the namespace:

            >>> index.documents.delete(
            ...     namespace="articles-en",
            ...     delete_all=True,
            ... )
        """
        require_non_empty("namespace", namespace)
        if ids is None and not delete_all and filter is None:
            raise PineconeValueError(
                "at least one of ids, delete_all=True, or filter must be provided"
            )
        if ids is not None and delete_all:
            raise PineconeValueError("ids and delete_all are mutually exclusive")
        if ids is not None and filter is not None:
            raise PineconeValueError("ids and filter are mutually exclusive")

        body: dict[str, Any] = {}
        if ids is not None:
            body["ids"] = ids
        if delete_all:
            body["delete_all"] = True
        if filter is not None:
            body["filter"] = filter

        self._http.post(
            f"/namespaces/{namespace}/documents/delete",
            json=body,
        )

    def __repr__(self) -> str:
        return "PreviewDocuments()"
