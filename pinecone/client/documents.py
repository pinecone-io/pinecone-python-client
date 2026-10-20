"""Documents namespace — document data-plane operations."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.documents_adapter import DocumentsAdapter
from pinecone._internal.bulk import bulk_execute_sync
from pinecone._internal.constants import DEFAULT_MAX_CONCURRENCY
from pinecone._internal.documents_helpers import (
    _build_delete_documents_body,
    _build_fetch_documents_body,
    _build_list_documents_body,
    _build_search_documents_body,
    _build_update_documents_body,
    _encode_document_namespace,
    _validate_documents,
)
from pinecone._internal.keyword_only import keyword_only_methods
from pinecone._internal.validation import require_in_range, validate_timeout
from pinecone.models.batch import BatchResult
from pinecone.models.documents.document import DocumentRecord, UpdateDocumentRecord
from pinecone.models.documents.responses import (
    DeleteDocumentsResponse,
    FetchDocumentsResponse,
    ListedDocumentRecord,
    SearchDocumentsResponse,
    UpdateDocumentsResponse,
    UpsertDocumentsResponse,
)
from pinecone.models.documents.score_by import DocumentScoringMethod
from pinecone.models.pagination import Page, Paginator

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)


@keyword_only_methods
class Documents:
    """Document data-plane operations for a schema-based index.

    A schema-based index stores JSON documents instead of raw vectors. Every
    document carries the reserved ``_id`` key; every other key is a field of
    your own, either declared in the index schema or free-form metadata.
    Accessed via :attr:`~pinecone.index.Index.documents`. Not constructed
    directly — the parent :class:`~pinecone.index.Index` builds and caches
    its own instance on first access.

    On a vector-based index, use the vector methods on
    :class:`~pinecone.index.Index` itself
    (:meth:`~pinecone.index.Index.upsert`,
    :meth:`~pinecone.index.Index.query`) rather than this namespace.
    Every method here is keyword-only. A positional argument raises
    :exc:`PineconeValueError` listing the accepted keywords, and a misspelled
    keyword raises :exc:`TypeError` suggesting the one you meant.

    Examples:
        On an index whose schema declares ``embedding`` as a 1536-dimension
        ``dense_vector`` and ``title`` as a full-text ``string``:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")
            with pc.index(name="articles-en") as idx:
                idx.documents.upsert(
                    namespace="published",
                    documents=[
                        {
                            "_id": "article-101",
                            "embedding": [0.1] * 1536,
                            "title": "Intro to vectors",
                        }
                    ],
                )

    .. seealso::
       `Error handling <https://sdk.pinecone.io/python/guides/error-handling.html>`__
       — the exceptions any of these methods can
       raise, and which ones are worth retrying.
    """

    def __init__(
        self,
        *,
        http: HTTPClient,
        host: str,
    ) -> None:
        self._http = http
        self._host = host

    def __repr__(self) -> str:
        return "Documents()"

    def upsert(
        self,
        *,
        namespace: str,
        documents: Sequence[Mapping[str, Any] | DocumentRecord],
        timeout: float | None = None,
    ) -> UpsertDocumentsResponse:
        """Upsert documents into a namespace.

        Each document must include an ``_id`` field (a unique, non-empty
        ASCII string of at most 512 characters). If a document with the same
        ``_id`` already exists in the namespace, it is overwritten.

        The index schema decides which other fields a document must carry;
        read it with ``pc.indexes.describe(name).schema``. Every
        ``dense_vector`` field is required, and so is every ``string`` field
        declared with ``embed``, because Pinecone embeds its text on write.
        ``sparse_vector`` fields, ``string`` fields with only ``sparse_embed``
        or full-text search, and metadata are optional. Each document also
        needs at least one searchable schema field: a vector field, or a
        ``string`` field with full-text search, ``embed`` or ``sparse_embed``.
        The server rejects a document that breaks either rule with a 400 that
        names the document and the fields it needs.

        Pinecone applies the upsert asynchronously, so documents may not be
        immediately visible to :meth:`search` or :meth:`fetch`.

        Args:
            namespace (str): Target namespace (required, non-empty).
            documents: The documents to upsert (1-1000 per request). Each
                element is a dict with an ``_id`` key or a
                :class:`~pinecone.models.documents.document.DocumentRecord`.
                On an index with an ``embed`` or ``sparse_embed`` field, the
                server takes at most 96 documents per request.
                A request is also limited by its encoded size, so documents
                that carry large fields, such as high-dimensional dense
                vectors, fit in fewer than 1000 per request. For larger lists,
                or documents of unknown size, use :meth:`batch_upsert`, which
                handles the size limit.
            timeout (float | None): Per-request timeout in seconds. Overrides
                the client-level default for this call only. Must be greater
                than zero.

        Returns:
            :class:`UpsertDocumentsResponse` with the count of documents
            accepted for upsert. ``usage.embed_total_tokens`` counts the tokens
            Pinecone embedded for ``embed`` and ``sparse_embed`` fields;
            ``usage`` is ``None`` when the upsert embedded no text.

        Raises:
            :exc:`PineconeValueError`: If ``namespace`` is empty, ``documents``
                is empty or over 1000 entries, or any document has a missing,
                empty, non-string, non-ASCII, over-512-character, or duplicate
                ``_id`` — the message names the offending document's position.
                Also raised if ``timeout`` is zero or negative.
            :exc:`ApiError`: If a document lacks a field its schema requires
                (400).
            :exc:`~pinecone.errors.exceptions.PayloadTooLargeError`: If one
                request exceeds the server's cap on encoded request size — a
                document count inside the accepted range can still be too
                large. Send fewer documents per request, or use
                :meth:`batch_upsert`.

        Examples:
            On an index whose schema declares ``embedding`` as a
            1536-dimension ``dense_vector`` and ``title`` as a full-text
            ``string``, every document carries ``embedding`` and ``title`` is
            optional:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> idx = pc.index(name="articles-en")
            >>> response = idx.documents.upsert(
            ...     namespace="published",
            ...     documents=[
            ...         {"_id": "article-101", "embedding": [0.1] * 1536, "title": "Vector basics"},
            ...         {"_id": "article-102", "embedding": [0.2] * 1536},
            ...     ],
            ... )
            >>> response.upserted_count
            2

        .. seealso::
           - :meth:`batch_upsert` — for upserting large document
             lists in parallel batches.
           - :meth:`~pinecone.index.Index.upsert` — for indexes where you
             provide your own vectors.
        """
        validate_timeout("timeout", timeout)
        segment = _encode_document_namespace(namespace)
        normalized = _validate_documents(documents)
        logger.info("Upserting %d documents into namespace %r", len(normalized), namespace)
        response = self._http.post(
            f"/namespaces/{segment}/documents/upsert",
            timeout=timeout,
            json={"documents": normalized},
        )
        return DocumentsAdapter.to_upsert_response(response)

    def batch_upsert(
        self,
        *,
        namespace: str,
        documents: Sequence[Mapping[str, Any] | DocumentRecord],
        batch_size: int = 50,
        max_concurrency: int | None = None,
        show_progress: bool = True,
        timeout: float | None = None,
        total_timeout: float | None = None,
    ) -> BatchResult:
        """Upsert a large list of documents in parallel batches.

        Splits *documents* into chunks of *batch_size* and submits them
        through the host's admission gate. Concurrency is bounded by
        *max_concurrency* and by the host's adaptive concurrency limit,
        whichever is lower, so a struggling backend applies backpressure
        instead of being handed every batch at once. Per-batch HTTP failures
        are captured in the returned
        :class:`~pinecone.models.batch.BatchResult` rather than raised, so one
        failed batch does not abort the rest; retry only the failures by
        passing ``result.failed_items`` back in.

        Requests are limited by encoded size as well as document count. When
        the server rejects a batch as too large (HTTP 413), the batch is
        halved and each half resent, recursively, so documents of any size
        are sent without choosing a *batch_size* by trial and error. A single
        document that is itself over the limit cannot be split: it is
        reported in ``result.failed_items`` with ``retryable=False``. A split
        batch can land in part: ``failed_items`` lists only the pieces that
        were rejected, and several ``result.errors`` rows can share one
        ``batch_index``.

        Args:
            namespace (str): Target namespace (required, non-empty).
            documents: Documents to upsert. Each element is a dict with an
                ``_id`` key or a :class:`~pinecone.models.documents.document.DocumentRecord`;
                IDs must be unique across the whole list. Each document must
                also carry every ``dense_vector`` field and every ``string``
                field declared with ``embed``; ``sparse_vector`` fields,
                ``sparse_embed``-only or full-text-only ``string`` fields, and
                metadata are optional, but each document needs at least one
                searchable schema field (see :meth:`upsert`). One document that
                breaks these rules fails its whole batch with a 400, recorded
                in ``result.errors`` with ``retryable=False``; fix the
                documents before resending.
            batch_size (int): Maximum documents per request (1-1000, default 50).
                On an index with an ``embed`` or ``sparse_embed`` field, the
                server takes at most 96 documents per request, so keep
                *batch_size* at 96 or below there. Requests are also limited
                by encoded size; a batch the server rejects as too large is
                halved and resent, so on any other index a *batch_size* above
                what fits by size is safe, at the cost of the rejected
                requests.
            max_concurrency (int | None): Upper bound on concurrent requests
                (1-64). Defaults to ``None``, which lets the admission gate
                use ``DEFAULT_MAX_CONCURRENCY`` (8); the gate's own adaptive
                limit for the host applies on top of whatever is passed.
            show_progress (bool): Display a progress bar when ``tqdm`` is
                installed. Defaults to ``True``.
            timeout (float | None): Per-request timeout in seconds applied to
                each batch's request — not to the whole call. Must be greater
                than zero.
            total_timeout (float | None): Deadline in seconds for the **whole
                batched upsert**, as opposed to *timeout*, which bounds a
                single attempt of a single batch. On expiry no further batches
                are submitted, and the un-submitted ones are reported in
                ``result.failed_items`` so they can be retried. ``None``
                (default) means no deadline; zero and negative values raise
                :exc:`PineconeValueError`. See the note below for what
                ``result.timed_out`` does and does not tell you.

        Returns:
            :class:`~pinecone.models.batch.BatchResult` with aggregated
            success and failure counts; per-batch errors are in
            ``result.errors`` and the affected documents in
            ``result.failed_items``.

        Raises:
            :exc:`PineconeValueError`: If ``namespace`` is empty, ``documents``
                is empty or contains an invalid or duplicate ``_id``,
                ``batch_size`` is outside [1, 1000], or ``max_concurrency`` is
                outside [1, 64].
                Also raised if ``timeout`` or ``total_timeout`` is zero or negative.

        Per-batch server rejections are not raised; each is recorded in
        ``result.errors`` as that batch's ``error``. A batch rejected as too
        large is halved and resent before anything is recorded, so a
        :exc:`~pinecone.errors.exceptions.PayloadTooLargeError` in
        ``result.errors`` belongs to a single document over the server's size
        cap. Resending it, or lowering *batch_size*, fails the same way;
        shrink that document instead.

        Examples:
            On an index whose schema declares ``embedding`` as a
            ``dense_vector`` and ``title`` as a full-text ``string``, where
            ``vectors`` holds one embedding per article:

            .. code-block:: python

                documents = [
                    {"_id": f"article-{i}", "embedding": vector, "title": f"Article {i}"}
                    for i, vector in enumerate(vectors)
                ]
                result = idx.documents.batch_upsert(
                    namespace="published",
                    documents=documents,
                    batch_size=100,
                    max_concurrency=8,
                    total_timeout=60.0,
                )
                print(result.successful_item_count, result.failed_item_count)
                if result.timed_out:
                    result = idx.documents.batch_upsert(
                        namespace="published",
                        documents=result.failed_items,
                        batch_size=100,
                    )

        .. note::
           Batches already in flight when *total_timeout* expires are awaited
           and never cancelled, because dropping one client-side would not
           stop the host from applying it. So ``result.timed_out`` is ``True``
           only when something was actually left unsent — if the in-flight
           batches were the last ones and all landed, the upsert succeeded
           late rather than failing. Time spent waiting for the host's
           admission gate counts against the budget, so a throttled host can
           consume it without a request being sent.

        .. seealso::
           - :meth:`upsert` — for a single-request upsert of up to
             1000 documents.
           - `How bulk ingest behaves <https://sdk.pinecone.io/python/guides/bulk-ingest.html>`__
             — choosing a batch size and
             concurrency, and reading the gate counters on the result.
        """
        validate_timeout("timeout", timeout)
        effective_max_concurrency = (
            DEFAULT_MAX_CONCURRENCY if max_concurrency is None else max_concurrency
        )
        require_in_range("batch_size", batch_size, 1, 1000)
        require_in_range("max_concurrency", effective_max_concurrency, 1, 64)
        segment = _encode_document_namespace(namespace)
        normalized = _validate_documents(documents, max_documents=None)

        def _operation(chunk: list[dict[str, Any]]) -> UpsertDocumentsResponse:
            response = self._http.post(
                f"/namespaces/{segment}/documents/upsert",
                timeout=timeout,
                json={"documents": chunk},
            )
            return DocumentsAdapter.to_upsert_response(response)

        return bulk_execute_sync(
            items=normalized,
            operation=_operation,
            batch_size=batch_size,
            max_concurrency=effective_max_concurrency,
            show_progress=show_progress,
            desc="Upserting documents",
            host=self._host,
            total_timeout=total_timeout,
            split_on_payload_too_large=True,
        )

    def search(
        self,
        *,
        namespace: str,
        score_by: Sequence[DocumentScoringMethod | Mapping[str, Any]],
        top_k: int,
        include_fields: Sequence[str] | None = None,
        filter: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> SearchDocumentsResponse:
        """Search documents in a namespace using one or more scoring methods.

        Returns the ``top_k`` most similar documents ranked by the given
        scoring methods (dense vector, sparse vector, BM25 text, or Lucene
        query string similarity, or similarity to query text that Pinecone
        embeds with the model of a field declared with ``embed`` or
        ``sparse_embed``).

        Args:
            namespace (str): Namespace to search (required, non-empty).
            score_by: Scoring methods to rank documents by (1-100 clauses),
                as a list. Wrap a single clause as ``[clause]``.
                Items are typed variants
                (:class:`~pinecone.models.documents.score_by.TextQuery`,
                :class:`~pinecone.models.documents.score_by.QueryStringQuery`,
                :class:`~pinecone.models.documents.score_by.DenseVectorQuery`,
                :class:`~pinecone.models.documents.score_by.SparseVectorQuery`,
                :class:`~pinecone.models.documents.score_by.EmbedQuery`,
                :class:`~pinecone.models.documents.score_by.SparseEmbedQuery`)
                or plain dicts with a ``type`` key. A dict clause with a key
                its type does not define, such as ``boost``, is rejected with a
                :exc:`PineconeValueError` rather than dropped. ``text`` and
                ``query_string`` clauses may be combined; a ``dense_vector``,
                ``sparse_vector``, ``embed`` or ``sparse_embed`` clause must
                appear alone. Combined clauses are OR'd, not AND'd: a document
                that matches any clause is returned, and one that matches
                several scores the sum of the scores from each clause it
                matches. A ``text`` clause naming several ``fields`` works the
                same way across those fields. Repeating a clause, naming a
                field twice, or writing one query as both a ``text`` and a
                ``query_string`` clause has been observed to score once (API
                2026-07), so repeating a clause does not up-weight it. To
                weight a term, give it a ``^N`` boost in a
                :class:`~pinecone.models.documents.score_by.QueryStringQuery`,
                such as ``title:entropy^5``; a ``text`` clause has no boost. To
                require several terms together, use ``AND`` inside one
                ``QueryStringQuery``.
            top_k (int): Number of top-ranked documents to return (1-10000).
            include_fields: Document fields to include in each match.
                Omitting it (the default) or passing ``[]`` returns only
                ``_id`` and ``_score``; ``["*"]`` returns every field, even
                alongside other names. :meth:`fetch` is the opposite — there,
                omitting the argument returns every field. Naming a field
                declared with ``store_text: false`` is rejected by the server
                with a ``400``, since its text is not stored; ``["*"]`` leaves
                it out.
            filter: Filter expression restricting the documents searched, or
                ``None``.
                Which operators a predicate may use depends on the field it names:

                - A data field, one the schema declares to be searched (a vector field, or
                  a string declared with ``full_text_search``, ``embed`` or
                  ``sparse_embed``), rejects comparison operators such as ``$eq`` with a
                  ``400``. Of these, a string whose text is stored and searchable takes
                  the text-match operators: one declared with ``full_text_search``, or
                  with ``embed`` or ``sparse_embed`` and the default ``store_text``
                  (the server turns full-text search on for it). A ``store_text=False``
                  string does not, and neither does a vector field. The operators are
                  ``$match_phrase`` (the words as a contiguous, ordered phrase),
                  ``$match_all`` (every word, in any order) and ``$match_any`` (at least
                  one word).
                - A metadata field, one the schema does not declare or declares only for
                  filtering (an integer, float, boolean, string list, or a string with no
                  search configuration), takes the comparison, set and logical operators
                  (``$eq``, ``$in``, ``$gte`` and so on). A text-match operator on it is
                  rejected with a ``400``.

                The two kinds combine under ``$and``.
            timeout (float | None): Per-request timeout in seconds. Overrides
                the client-level default for this call only. Must be greater
                than zero.

        Returns:
            :class:`SearchDocumentsResponse` with ``matches`` (ordered from
            most to least similar), ``namespace``, and ``usage``, whose
            ``embed_total_tokens`` counts the query tokens Pinecone embedded
            when the server reports it. Each match
            is a :class:`~pinecone.models.documents.document.Document`,
            reached as ``doc.id``, ``doc.score``, and ``doc.<field>`` for the
            fields ``include_fields`` asked for.

        Raises:
            :exc:`PineconeValueError`: If ``namespace`` is empty, ``score_by``
                is empty, over 100 clauses, or combines a vector, ``embed``
                or ``sparse_embed`` clause with any other clause, or ``top_k``
                is outside [1, 10000].
                Also raised if ``timeout`` is zero or negative.
            :exc:`PineconeTypeError`: If ``top_k`` is not an integer (a bool is
                not accepted), ``score_by`` is not a list of clauses (wrap a
                single clause as ``[clause]``), holds an item that is neither a
                dict nor one of the six clause types (such as a
                :class:`~pinecone.models.vectors.search.SearchQuery`),
                ``filter`` is not a dict (call ``.to_dict()`` on a
                filter-builder :class:`~pinecone.utils.filter_builder.Condition`),
                or ``include_fields`` is a bare ``str`` rather than a list.

        Examples:
            >>> from pinecone import Pinecone, TextQuery
            >>> pc = Pinecone(api_key="your-api-key")
            >>> idx = pc.index(name="articles-en")
            >>> response = idx.documents.search(
            ...     namespace="published",
            ...     top_k=5,
            ...     score_by=[TextQuery(query="machine learning", fields=["content"])],
            ...     include_fields=["title", "content"],
            ...     filter={"category": {"$eq": "tech"}},
            ... )
            >>> for doc in response.matches:
            ...     print(doc.id, doc.score)
            article-101 0.92
            article-102 0.74

            Filter on a schema field with a text-match operator (here ``title``
            is a string field declared with ``full_text_search``):

            >>> response = idx.documents.search(
            ...     namespace="published",
            ...     top_k=5,
            ...     score_by=[TextQuery(query="machine learning", fields=["content"])],
            ...     filter={"title": {"$match_phrase": "machine learning"}},
            ... )
            >>> for doc in response.matches:
            ...     print(doc.id, doc.score)
            article-101 0.92
            article-102 0.74

        .. seealso::
           - :meth:`~pinecone.index.Index.search` — record search for
             legacy integrated indexes, through the legacy Records API.
           - :meth:`~pinecone.index.Index.query` — nearest-neighbor search
             over vectors you provide.
        """
        validate_timeout("timeout", timeout)
        segment = _encode_document_namespace(namespace)
        body = _build_search_documents_body(
            score_by=score_by,
            top_k=top_k,
            include_fields=include_fields,
            filter=filter,
        )
        logger.info("Searching documents in namespace %r with top_k=%d", namespace, top_k)
        response = self._http.post(
            f"/namespaces/{segment}/documents/search",
            timeout=timeout,
            json=body,
        )
        return DocumentsAdapter.to_search_response(response)

    def fetch(
        self,
        *,
        namespace: str,
        ids: Sequence[str] | None = None,
        filter: Mapping[str, Any] | None = None,
        include_fields: Sequence[str] | None = None,
        pagination_token: str | None = None,
        limit: int | None = None,
        timeout: float | None = None,
    ) -> FetchDocumentsResponse:
        """Fetch documents from a namespace by ID or by metadata filter.

        Exactly one of ``ids`` or ``filter`` must be provided. A filtered
        fetch returns matching documents a page at a time, with
        ``response.pagination`` carrying the token for the next page.

        Args:
            namespace (str): Namespace to fetch from (required, non-empty).
            ids: Document IDs to fetch (1-1000). IDs that do not exist are
                omitted from the result rather than raising. Mutually
                exclusive with ``filter``.
            filter: Non-empty filter expression selecting the documents to
                fetch. Mutually exclusive with ``ids``.
                Which operators a predicate may use depends on the field it names:

                - A data field, one the schema declares to be searched (a vector field, or
                  a string declared with ``full_text_search``, ``embed`` or
                  ``sparse_embed``), rejects comparison operators such as ``$eq`` with a
                  ``400``. Of these, a string whose text is stored and searchable takes
                  the text-match operators: one declared with ``full_text_search``, or
                  with ``embed`` or ``sparse_embed`` and the default ``store_text``
                  (the server turns full-text search on for it). A ``store_text=False``
                  string does not, and neither does a vector field. The operators are
                  ``$match_phrase`` (the words as a contiguous, ordered phrase),
                  ``$match_all`` (every word, in any order) and ``$match_any`` (at least
                  one word).
                - A metadata field, one the schema does not declare or declares only for
                  filtering (an integer, float, boolean, string list, or a string with no
                  search configuration), takes the comparison, set and logical operators
                  (``$eq``, ``$in``, ``$gte`` and so on). A text-match operator on it is
                  rejected with a ``400``.

                The two kinds combine under ``$and``.
            include_fields: Document fields to include in each document.
                Omitting it (the default), ``[]``, or ``["*"]`` each return
                every field; a list of names returns just those.
                :meth:`search` is the opposite — there, omitting the argument
                returns only ``_id`` and ``_score``. Every field means every
                field the index stores: a field declared with ``store_text:
                false`` is left out, and naming one is rejected by the server
                with a ``400``.
            pagination_token: Token from a previous filtered fetch response
                to retrieve the next page. Only valid together with
                ``filter``.
            limit: Maximum number of documents per page of a filtered fetch
                (1-10000). Omitting it uses the server default of 100. A fetch
                by ``ids`` ignores it.
            timeout (float | None): Per-request timeout in seconds. Overrides
                the client-level default for this call only. Must be greater
                than zero.

        Returns:
            :class:`FetchDocumentsResponse` with ``documents`` (document ID
            mapped to a :class:`~pinecone.models.documents.document.Document`,
            reached as ``doc.<field>``), ``namespace``, ``usage``, and — for
            filtered fetches with more results — ``pagination``.

        Raises:
            :exc:`PineconeValueError`: If ``namespace`` is empty, both or
                neither of ``ids`` and ``filter`` are provided, ``filter`` is
                an empty dict, ``ids`` exceeds 1000 entries,
                ``pagination_token`` is passed without ``filter``, or
                ``limit`` is outside 1-10000.
                Also raised if ``timeout`` is zero or negative.
            :exc:`PineconeTypeError`: If ``limit`` is not an integer (a bool is
                not accepted), ``filter`` is not a dict (call
                ``.to_dict()`` on a filter-builder
                :class:`~pinecone.utils.filter_builder.Condition`), or
                ``ids`` or ``include_fields`` is a bare ``str`` rather than
                a list.

        Examples:
            Fetch specific documents by ID. IDs that do not exist are absent
            from ``response.documents`` rather than raising:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> idx = pc.index(name="articles-en")
            >>> response = idx.documents.fetch(
            ...     namespace="published",
            ...     ids=["article-101", "article-102"],
            ... )
            >>> for doc_id, doc in response.documents.items():
            ...     print(doc_id, doc.title)
            article-101 Intro to vectors
            article-102 Advanced retrieval

            Fetch by filter instead. A filtered fetch is paginated, so read
            each page's documents before asking for the next one — the loop
            below is the whole retrieval, not just the token bookkeeping:

            >>> pagination_token = None
            >>> while True:
            ...     response = idx.documents.fetch(
            ...         namespace="published",
            ...         filter={"category": {"$eq": "tech"}},
            ...         pagination_token=pagination_token,
            ...     )
            ...     for doc_id, doc in response.documents.items():
            ...         print(doc_id, doc.title)
            ...     if response.pagination is None:
            ...         break
            ...     pagination_token = response.pagination.next
            article-101 Intro to vectors
            article-102 Advanced retrieval

            A filter can use a text-match operator on a schema field, here
            ``title``, a string field declared with ``full_text_search``,
            instead of a comparison on a metadata field:

            >>> response = idx.documents.fetch(
            ...     namespace="published",
            ...     filter={"title": {"$match_any": "vectors retrieval"}},
            ... )
            >>> for doc_id, doc in response.documents.items():
            ...     print(doc_id, doc.title)
            article-101 Intro to vectors
            article-102 Advanced retrieval

        .. seealso::
           - :meth:`search` — when you want the best-matching documents
             rather than every document that satisfies a filter.
           - :meth:`~pinecone.index.Index.fetch` — for indexes where you
             provide your own vectors.
           - `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__
             — the pagination shapes the SDK uses
             and when each applies.
        """
        validate_timeout("timeout", timeout)
        segment = _encode_document_namespace(namespace)
        body = _build_fetch_documents_body(
            ids=ids,
            filter=filter,
            include_fields=include_fields,
            pagination_token=pagination_token,
            limit=limit,
        )
        logger.info("Fetching documents from namespace %r", namespace)
        response = self._http.post(
            f"/namespaces/{segment}/documents/fetch",
            timeout=timeout,
            json=body,
        )
        return DocumentsAdapter.to_fetch_response(response)

    def delete(
        self,
        *,
        namespace: str,
        ids: Sequence[str] | None = None,
        filter: Mapping[str, Any] | None = None,
        delete_all: bool = False,
        timeout: float | None = None,
    ) -> DeleteDocumentsResponse:
        """Delete documents from a namespace by ID, filter, or delete-all flag.

        Exactly one of ``ids``, ``filter``, or ``delete_all`` must be
        provided. Deleting IDs that do not exist does not raise an error.

        Pinecone applies the delete asynchronously. For a filtered delete,
        ``response.matched_records`` is the point-in-time count of matching
        documents when the delete was accepted, not a guarantee of the number
        ultimately deleted.

        Args:
            namespace (str): Namespace to delete from (required, non-empty).
            ids: Document IDs to delete (1-1000). Mutually exclusive with
                ``filter`` and ``delete_all``.
            filter: Non-empty metadata filter expression selecting the
                documents to delete. Text-match operators (``$match_phrase``,
                ``$match_all``, ``$match_any``) are not supported here.
                Mutually exclusive with ``ids`` and ``delete_all``. Not
                available on an index with dedicated read capacity scaled to
                0 replicas — scale up replicas first.
                Pass a dict; for a filter-builder ``Condition``, pass
                ``condition.to_dict()``.
            delete_all (bool): If ``True``, delete all documents in the
                namespace. Mutually exclusive with ``ids`` and ``filter``.
            timeout (float | None): Per-request timeout in seconds. Overrides
                the client-level default for this call only. Must be greater
                than zero.

        Returns:
            :class:`DeleteDocumentsResponse` — ``matched_records`` is
            populated only for filtered deletes (``None`` for by-ID and
            delete-all paths, and when the count could not be read in time).

        Raises:
            :exc:`PineconeValueError`: If ``namespace`` is empty, zero or more
                than one of ``ids``/``filter``/``delete_all`` is provided,
                ``filter`` is an empty dict, or ``ids`` exceeds 1000 entries.
                Also raised if ``timeout`` is zero or negative.
            :exc:`PineconeTypeError`: If ``filter`` is not a dict (call
                ``.to_dict()`` on a filter-builder
                :class:`~pinecone.utils.filter_builder.Condition`), or
                ``ids`` is a bare ``str`` rather than a list.

        Examples:
            Delete specific documents by ID:

            .. code-block:: python

                idx.documents.delete(namespace="published", ids=["article-101"])

            Delete every document matching a filter:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> idx = pc.index(name="articles-en")
            >>> response = idx.documents.delete(
            ...     namespace="published",
            ...     filter={"category": {"$eq": "obsolete"}},
            ... )
            >>> response.matched_records
            2

            Or empty a namespace outright. ``delete_all`` removes every
            document in the namespace named — it takes no ``ids`` or
            ``filter`` to narrow it:

            .. code-block:: python

                idx.documents.delete(namespace="drafts", delete_all=True)

        .. seealso::
           - :meth:`~pinecone.index.Index.delete_namespace` — to remove the
             namespace itself, rather than emptying it with ``delete_all``.
           - :meth:`~pinecone.index.Index.delete` — for indexes where you
             provide your own vectors.
        """
        validate_timeout("timeout", timeout)
        segment = _encode_document_namespace(namespace)
        body = _build_delete_documents_body(ids=ids, filter=filter, delete_all=delete_all)
        logger.info("Deleting documents from namespace %r", namespace)
        response = self._http.post(
            f"/namespaces/{segment}/documents/delete",
            timeout=timeout,
            json=body,
        )
        return DocumentsAdapter.to_delete_response(response)

    def update(
        self,
        *,
        namespace: str,
        documents: Sequence[Mapping[str, Any] | UpdateDocumentRecord] | None = None,
        filter: Mapping[str, Any] | None = None,
        set_fields: Mapping[str, Any] | None = None,
        remove_fields: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> UpdateDocumentsResponse:
        """Apply partial updates to documents in a namespace.

        Documents are selected either per ID with ``documents``, or in bulk
        with ``filter`` plus ``set_fields`` and/or ``remove_fields``. The two
        shapes are mutually exclusive. Fields that are not mentioned are left
        unchanged, and an update naming a document that does not exist is
        accepted as a no-op rather than raising.

        Pinecone applies the update asynchronously — for a filtered update,
        ``response.matched_records`` is the point-in-time count of matching
        documents when the update was accepted, not a guarantee of the number
        ultimately patched.

        Args:
            namespace (str): Namespace to update in (required, non-empty).
            documents: Per-document patches (1-1000, or at most 96 on an
                index with an ``embed`` or ``sparse_embed`` field). Each element
                is a dict with an ``_id`` key or an
                :class:`~pinecone.models.documents.document.UpdateDocumentRecord`.
                Any key other than ``_id`` and ``_remove_fields`` sets a new
                value for that field; the names in ``_remove_fields`` are
                removed from the document. ``_id`` values must be unique
                within the request. Mutually exclusive with ``filter``,
                ``set_fields``, and ``remove_fields``.
            filter: Non-empty metadata filter expression selecting the
                documents to patch. Text-match operators (``$match_phrase``,
                ``$match_all``, ``$match_any``) are not supported here.
                Mutually exclusive with ``documents``. Not available on an
                index with dedicated read capacity scaled to 0 replicas —
                scale up replicas first.
                Pass a dict; for a filter-builder ``Condition``, pass
                ``condition.to_dict()``.
            set_fields: Fields to set on every document matching ``filter``,
                and the values to set them to. Only valid with ``filter``.
            remove_fields: Names of the fields to remove from every document
                matching ``filter``. Only valid with ``filter``.
            timeout (float | None): Per-request timeout in seconds. Overrides
                the client-level default for this call only. Must be greater
                than zero.

        Returns:
            :class:`UpdateDocumentsResponse` — ``matched_records`` is
            populated only for filtered updates (``None`` for per-ID updates,
            and when the count could not be read in time).
            ``usage.embed_total_tokens`` counts the tokens Pinecone embedded
            for ``embed`` and ``sparse_embed`` fields; ``usage`` is ``None``
            when the update embedded no text.

        Raises:
            :exc:`PineconeValueError`: If ``namespace`` is empty,
                ``documents`` is combined with any by-filter field, neither
                ``documents`` nor ``filter`` is given, ``set_fields`` or
                ``remove_fields`` is passed without ``filter``, ``filter`` is
                an empty dict or carries no patch, ``documents`` is empty or
                exceeds 1000 entries, or a patch is malformed — the message
                names the offending position.
                Also raised if ``timeout`` is zero or negative.
            :exc:`ApiError`: If a field value is ``None``, which the server
                rejects — use ``_remove_fields`` (per-ID) or ``remove_fields``
                (by-filter) to remove a field instead.
            :exc:`PineconeTypeError`: If ``filter`` is not a dict (call
                ``.to_dict()`` on a filter-builder
                :class:`~pinecone.utils.filter_builder.Condition`), or
                ``remove_fields`` is a bare ``str`` rather than a list.

        Examples:
            Patch documents by ID. Each key other than the reserved ``_id`` and
            ``_remove_fields`` sets that field's value; fields the patch does
            not name keep the values they already have, so ``article-101`` here
            gets a new ``title`` and is otherwise untouched:

            .. code-block:: python

                idx.documents.update(
                    namespace="published",
                    documents=[
                        {"_id": "article-101", "title": "An introduction to vector search"},
                        {"_id": "article-102", "_remove_fields": ["draft_notes"]},
                    ],
                )

            ``article-102`` keeps every field it has except ``draft_notes``:
            ``_remove_fields`` names fields to delete rather than setting a
            field called ``_remove_fields``.

            Patch every document matching a filter instead, setting one field
            and removing another across all of them:

            .. code-block:: python

                response = idx.documents.update(
                    namespace="published",
                    filter={"category": {"$eq": "news"}},
                    set_fields={"review_status": "archived"},
                    remove_fields=["draft_notes"],
                )
                print(response.matched_records)

        .. seealso::
           - :meth:`upsert` — to replace a document outright; an upsert of an
             existing ``_id`` drops the fields it does not mention, where this
             method keeps them.
           - :meth:`~pinecone.index.Index.update` — for indexes where you
             provide your own vectors.
        """
        validate_timeout("timeout", timeout)
        segment = _encode_document_namespace(namespace)
        body = _build_update_documents_body(
            documents=documents,
            filter=filter,
            set_fields=set_fields,
            remove_fields=remove_fields,
        )
        logger.info("Updating documents in namespace %r", namespace)
        response = self._http.post(
            f"/namespaces/{segment}/documents/update",
            timeout=timeout,
            json=body,
        )
        return DocumentsAdapter.to_update_response(response)

    def list(
        self,
        *,
        namespace: str,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        timeout: float | None = None,
    ) -> Paginator[ListedDocumentRecord]:
        """List the documents in a namespace, following pagination lazily.

        Returns a :class:`~pinecone.models.pagination.Paginator` that fetches
        pages on demand and stops when the server returns no ``pagination``
        token. Documents come back in sorted order by ID, carrying only their
        ``_id``.

        Args:
            namespace (str): Namespace to list from (required, non-empty).
            prefix (str | None): Return only documents whose IDs begin with
                this prefix. At most 512 characters, ASCII only
                (``\\x01``-``\\x7F``). ``None`` lists every document.
            limit (int | None): Maximum number of documents the server returns
                **per page**, 1-100. This tunes the page size, not the total —
                the paginator follows every page. ``None`` (default) lets the
                server choose the page size. To stop early, break out of the
                loop.
            pagination_token (str | None): Token from a previous list response
                to resume from, rather than starting at the first page.
            timeout (float | None): Per-request timeout in seconds, applied to
                each page request. Overrides the client-level default. Must be
                greater than zero.

        Returns:
            :class:`~pinecone.models.pagination.Paginator` over
            :class:`ListedDocumentRecord` objects. Supports ``for`` loops,
            :meth:`~pinecone.models.pagination.Paginator.to_list`, and
            :meth:`~pinecone.models.pagination.Paginator.pages`.

        Raises:
            :exc:`PineconeTypeError`: If ``limit`` is not an integer (a bool is
                not accepted).
            :exc:`PineconeValueError`: If ``namespace`` is empty, ``prefix``
                violates the rules above, or ``limit`` falls outside 1-100.
                Raised by this call, before the paginator is returned.
                Also raised if ``timeout`` is zero or negative.

        Examples:
            Iterate every document ID in the namespace, letting the paginator
            cross page boundaries for you:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> idx = pc.index(name="articles-en")
            >>> for doc in idx.documents.list(namespace="published", prefix="article-1"):
            ...     print(doc.id)
            article-101
            article-102

            Or take the pages themselves, when you want to checkpoint a long
            walk on the token each page carries:

            >>> for page in idx.documents.list(namespace="published", limit=20).pages():
            ...     print(len(page.items), page.pagination_token)
            2 None

        .. seealso::
           - :meth:`fetch` — to read the fields of the documents, not just
             their IDs.
           - :meth:`~pinecone.index.Index.list` — for indexes where you
             provide your own vectors.
           - `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__
             — the pagination shapes the SDK uses
             and when each applies.
        """
        validate_timeout("timeout", timeout)
        segment = _encode_document_namespace(namespace)
        base = _build_list_documents_body(prefix=prefix, limit=limit, pagination_token=None)

        def fetch_page(token: str | None) -> Page[ListedDocumentRecord]:
            body = dict(base)
            if token is not None:
                body["pagination_token"] = token
            logger.info("Listing documents in namespace %r", namespace)
            response = self._http.post(
                f"/namespaces/{segment}/documents/list",
                timeout=timeout,
                json=body,
            )
            result = DocumentsAdapter.to_list_response(response)
            next_token = result.pagination.next if result.pagination is not None else None
            return Page(items=result.documents, pagination_token=next_token)

        return Paginator(fetch_page=fetch_page, initial_token=pagination_token)
