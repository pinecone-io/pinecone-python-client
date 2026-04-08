"""Synchronous data plane client for a Pinecone index."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

from pinecone._internal.adapters.imports_adapter import ImportsAdapter
from pinecone._internal.adapters.vectors_adapter import VectorsAdapter, extract_response_info
from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import DATA_PLANE_API_VERSION
from pinecone._internal.data_plane_helpers import _validate_host, _vector_to_dict
from pinecone._internal.vector_factory import VectorFactory
from pinecone.errors.exceptions import PineconeValueError, ValidationError
from pinecone.models.imports.list import ImportList
from pinecone.models.imports.model import ImportModel, StartImportResponse
from pinecone.models.namespaces.models import ListNamespacesResponse, NamespaceDescription
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults, QueryResultsAggregator
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchByMetadataResponse,
    FetchResponse,
    ListResponse,
    QueryResponse,
    UpdateResponse,
    UpsertRecordsResponse,
    UpsertResponse,
)
from pinecone.models.vectors.search import RerankConfig, SearchRecordsResponse
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import Vector

logger = logging.getLogger(__name__)


class Index:
    """Synchronous data plane client targeting a specific Pinecone index.

    Can be constructed directly with a host URL, or via the
    :meth:`Pinecone.index` factory method.

    Args:
        host (str): The index-specific data plane host URL.
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        additional_headers (dict[str, str] | None): Extra headers included in every request.
        timeout (float): Request timeout in seconds. Defaults to ``30.0``.
        proxy_url (str | None): HTTP proxy URL for outgoing requests.
        ssl_ca_certs (str | None): Path to a CA certificate bundle for SSL verification.
        ssl_verify (bool): Whether to verify SSL certificates. Defaults to ``True``.
        source_tag (str | None): Tag appended to the User-Agent string for request attribution.
        connection_pool_maxsize (int): Maximum number of connections to keep in the pool.
            ``0`` (default) uses httpx defaults.

    Raises:
        :exc:`ValidationError`: If no API key can be resolved or the host is invalid.

    Examples:

        from pinecone import Index

        idx = Index(host="movie-recs-abc123.svc.pinecone.io", api_key="...")
    """

    def __init__(
        self,
        *,
        host: str,
        api_key: str | None = None,
        additional_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        proxy_url: str | None = None,
        ssl_ca_certs: str | None = None,
        ssl_verify: bool = True,
        source_tag: str | None = None,
        connection_pool_maxsize: int = 0,
    ) -> None:
        # Resolve API key: explicit arg > env var (check BEFORE host per unified-ord-0001)
        resolved_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        if not resolved_key:
            raise ValidationError(
                "No API key provided. Pass api_key='...' or set the "
                "PINECONE_API_KEY environment variable."
            )

        # Validate and normalize host
        self._host = _validate_host(host)

        config = PineconeConfig(
            api_key=resolved_key,
            host=self._host,
            timeout=timeout,
            additional_headers=additional_headers or {},
            proxy_url=proxy_url or "",
            ssl_ca_certs=ssl_ca_certs,
            ssl_verify=ssl_verify,
            source_tag=source_tag or "",
            connection_pool_maxsize=connection_pool_maxsize,
        )
        self._config = config

        from pinecone._internal.http_client import HTTPClient

        self._http = HTTPClient(config, DATA_PLANE_API_VERSION)
        self._adapter = VectorsAdapter()
        self._imports_adapter = ImportsAdapter()

        logger.info("Index client created for host %s", self._host)

    @property
    def host(self) -> str:
        """The data plane host URL for this index."""
        return self._host

    def upsert(
        self,
        *,
        vectors: Sequence[
            Vector
            | tuple[str, list[float]]
            | tuple[str, list[float], dict[str, Any]]
            | dict[str, Any]
        ],
        namespace: str = "",
    ) -> UpsertResponse:
        """Upsert a batch of vectors into a namespace.

        If a vector with the same ID already exists in the namespace, it is
        overwritten.

        Args:
            vectors: Sequence of vectors to upsert. Each element can be a
                ``Vector`` instance, a tuple of ``(id, values)`` or
                ``(id, values, metadata)``, or a dict with ``id``, ``values``,
                and optional ``sparse_values`` / ``metadata`` keys.
            namespace (str): Target namespace. Defaults to the default
                (empty-string) namespace.

        Returns:
            :class:`UpsertResponse` with the count of vectors upserted.

        Raises:
            :exc:`PineconeTypeError`: If a vector element is not a recognized format.
            :exc:`PineconeValueError`: If a vector element is malformed.
            :exc:`ApiError`: If the API returns an error response (e.g. authentication
                failure or server error).
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            from pinecone import Index, Vector

            idx = Index(host="article-search-abc123.svc.pinecone.io", api_key="...")
            response = idx.upsert(
                vectors=[
                    Vector(
                        id="article-101",
                        values=[0.012, -0.087, 0.153],  # truncated; use your actual dimension
                    ),
                    ("article-102", [0.045, 0.021, -0.064]),  # truncated
                    {"id": "article-103", "values": [0.091, -0.032, 0.178]},  # truncated
                ],
                namespace="articles-en",
            )
            print(response.upserted_count)

        .. note::
           All vectors are sent in a single request. For large datasets,
           batch your calls (recommended batch size: 100–500 vectors) or use
           :meth:`upsert_from_dataframe` which handles batching automatically.
           For very large datasets (millions of vectors), consider
           :meth:`start_import` for bulk import from cloud storage.

        .. seealso::
           - :meth:`upsert_records` — for indexes with integrated inference
             (text in, server-side embedding).
           - :meth:`upsert_from_dataframe` — for loading from a pandas
             DataFrame with automatic batching.
           - :meth:`start_import` — for bulk loading millions of vectors
             from cloud storage (S3, GCS).
        """
        built = [VectorFactory.build(v) for v in vectors]
        body: dict[str, Any] = {
            "vectors": [_vector_to_dict(v) for v in built],
        }
        if namespace:
            body["namespace"] = namespace

        logger.info("Upserting %d vectors into namespace %r", len(built), namespace)
        response = self._http.post("/vectors/upsert", json=body)
        result = self._adapter.to_upsert_response(response.content)
        result.response_info = extract_response_info(response)
        logger.debug("Upserted %d vectors", result.upserted_count)
        return result

    def upsert_from_dataframe(
        self,
        df: pd.DataFrame,
        namespace: str | None = None,
        batch_size: int = 500,
        show_progress: bool = True,
    ) -> UpsertResponse:
        """Upsert vectors from a pandas DataFrame.

        Convenience method that accepts a DataFrame with columns ``id``,
        ``values``, and optionally ``sparse_values`` and ``metadata``,
        batches the rows, and upserts them via :meth:`upsert`.

        Args:
            df: A ``pandas.DataFrame`` with at least ``id`` and ``values``
                columns. ``sparse_values`` and ``metadata`` columns are
                included when present and non-None.
            namespace: Target namespace. Defaults to the default namespace.
            batch_size: Number of rows per upsert batch. Defaults to 500.
            show_progress: If ``True`` and ``tqdm`` is installed, display a
                progress bar. If ``tqdm`` is not installed, silently falls
                back to no progress bar.

        Returns:
            :class:`UpsertResponse` with the total count of vectors upserted across
            all batches.

        Raises:
            :exc:`RuntimeError`: If ``pandas`` is not installed.
            :exc:`PineconeValueError`: If *df* is not a ``pandas.DataFrame``.
            :exc:`PineconeValueError`: If *batch_size* is not a positive integer.

        Examples:
            Upsert article embeddings from a DataFrame:

            >>> import pandas as pd
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> index = pc.Index("article-search")
            >>> df = pd.DataFrame([
            ...     {"id": "article-101", "values": [0.012, -0.087, 0.153, ...]},
            ...     {"id": "article-102", "values": [0.045, 0.021, -0.064, ...]},
            ... ])
            >>> response = index.upsert_from_dataframe(df)
            >>> response.upserted_count
            2

            Upsert with metadata, a custom namespace, and a smaller batch size:

            >>> df = pd.DataFrame([
            ...     {
            ...         "id": "article-101",
            ...         "values": [0.012, -0.087, 0.153, ...],
            ...         "metadata": {"topic": "science", "year": 2024},
            ...     },
            ...     {
            ...         "id": "article-102",
            ...         "values": [0.045, 0.021, -0.064, ...],
            ...         "metadata": {"topic": "technology", "year": 2024},
            ...     },
            ... ])
            >>> response = index.upsert_from_dataframe(
            ...     df,
            ...     namespace="articles-en",
            ...     batch_size=100,
            ... )

        .. seealso::
           - :meth:`upsert` — for upserting vectors directly (single batch,
             no DataFrame dependency).
           - :meth:`upsert_records` — for indexes with integrated inference
             (text in, server-side embedding).
           - :meth:`start_import` — for bulk loading millions of vectors
             from cloud storage (S3, GCS).
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError(
                "pandas is required for upsert_from_dataframe. Install it with: pip install pandas"
            )

        if not isinstance(df, pd.DataFrame):
            raise PineconeValueError("df must be a pandas DataFrame")

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise PineconeValueError("batch_size must be a positive integer")

        has_sparse = "sparse_values" in df.columns
        has_metadata = "metadata" in df.columns

        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            record: dict[str, Any] = {"id": row["id"], "values": row["values"]}
            if has_sparse and row["sparse_values"] is not None:
                record["sparse_values"] = row["sparse_values"]
            if has_metadata and row["metadata"] is not None:
                record["metadata"] = row["metadata"]
            records.append(record)

        batches: list[list[dict[str, Any]]] = [
            records[i : i + batch_size] for i in range(0, len(records), batch_size)
        ]

        batch_iter: Any = batches
        if show_progress:
            try:
                from tqdm.auto import tqdm  # type: ignore[import-untyped]

                batch_iter = tqdm(batches, desc="Upserting")
            except ImportError:
                pass

        total_count = 0
        ns = namespace or ""
        for batch in batch_iter:
            result = self.upsert(vectors=batch, namespace=ns)
            total_count += result.upserted_count

        return UpsertResponse(upserted_count=total_count)

    def upsert_records(
        self,
        *,
        records: list[dict[str, Any]],
        namespace: str,
    ) -> UpsertRecordsResponse:
        """Upsert records for indexes with integrated inference.

        Records are sent as newline-delimited JSON (NDJSON). Embeddings are
        generated server-side.

        Args:
            records: List of record dicts. Each must contain an ``_id`` or
                ``id`` field. Additional fields are passed through for
                server-side embedding.
            namespace (str): Target namespace (required).

        Returns:
            :class:`UpsertRecordsResponse` with the count of records submitted.

        Raises:
            :exc:`ValidationError`: If records is empty or a record is missing an
                identifier field.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            response = idx.upsert_records(
                namespace="articles-en",
                records=[
                    {"_id": "article-101", "text": "Vector databases enable similarity search."},
                    {"_id": "article-102", "text": "RAG combines search with LLMs."},
                ],
            )
            print(response.record_count)

        .. seealso::
           - :meth:`upsert` — for indexes where you provide your own vectors
             (no server-side embedding).
           - :meth:`upsert_from_dataframe` — for loading vectors from a
             pandas DataFrame with automatic batching.
           - :meth:`start_import` — for bulk loading millions of vectors
             from cloud storage (S3, GCS).
        """
        if not records:
            raise ValidationError("records must be a non-empty list")

        for i, record in enumerate(records):
            if "_id" not in record and "id" not in record:
                raise ValidationError(f"Record at index {i} must contain an '_id' or 'id' field")

        import orjson

        normalized: list[dict[str, Any]] = []
        for record in records:
            r = dict(record)  # shallow copy
            if "_id" not in r and "id" in r:
                r["_id"] = r.pop("id")
            normalized.append(r)

        ndjson_lines = [orjson.dumps(r).decode("utf-8") for r in normalized]
        ndjson_body = "\n".join(ndjson_lines) + "\n"

        logger.info("Upserting %d records into namespace %r (NDJSON)", len(records), namespace)
        response = self._http.post(
            f"/records/namespaces/{namespace}/upsert",
            content=ndjson_body.encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
        )
        result = UpsertRecordsResponse(record_count=len(records))
        result.response_info = extract_response_info(response)
        return result

    def query(
        self,
        *,
        top_k: int,
        vector: list[float] | None = None,
        id: str | None = None,
        namespace: str = "",
        filter: dict[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = False,
        sparse_vector: SparseValues | dict[str, Any] | None = None,
        scan_factor: float | None = None,
        max_candidates: int | None = None,
    ) -> QueryResponse:
        """Query a namespace for the nearest neighbors of a vector.

        .. note::
           Use this method for indexes where you provide your own vectors.
           For indexes with integrated inference (``IntegratedSpec``), use
           :meth:`search` which handles embedding server-side.

        Args:
            top_k (int): Number of results to return (must be >= 1).
            vector (list[float] | None): Dense query vector values.
            id (str | None): ID of a stored vector to use as the query.
            namespace (str): Namespace to query. Defaults to the default namespace.
            filter (dict[str, Any] | None): Metadata filter expression.
            include_values (bool): Whether to include vector values in results.
            include_metadata (bool): Whether to include metadata in results.
            sparse_vector (SparseValues | dict[str, Any] | None): Sparse query vector
                with indices and values.
            scan_factor (float | None): DRN optimization — adjusts how much of the
                index is scanned. Range 0.5–4.0. Only supported for dedicated read
                node indexes. None uses server default.
            max_candidates (int | None): DRN optimization — caps candidate vectors to
                rerank. Range 1–100000. Only supported for dedicated read node indexes.
                None uses server default.

        Returns:
            :class:`QueryResponse` with matches, namespace, and usage info.

        Raises:
            :exc:`ValidationError`: If top_k < 1, or both/neither vector and id provided.
            :exc:`ApiError`: If the API returns an error response (e.g. authentication
                failure or server error).
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            response = idx.query(
                top_k=10,
                vector=[0.012, -0.087, 0.153],  # truncated; use your actual dimension
            )
            for match in response.matches:
                print(match.id, match.score)
        """
        if top_k < 1:
            raise ValidationError(f"top_k must be a positive integer, got {top_k}")

        has_vector = vector is not None
        has_id = id is not None
        if has_vector and has_id:
            raise ValidationError("Exactly one of vector or id must be provided, not both")
        if not has_vector and not has_id:
            raise ValidationError("Exactly one of vector or id must be provided, got neither")

        body: dict[str, Any] = {
            "topK": top_k,
            "includeValues": include_values,
            "includeMetadata": include_metadata,
        }
        if namespace:
            body["namespace"] = namespace
        if vector is not None:
            body["vector"] = vector
        if id is not None:
            body["id"] = id
        if filter is not None:
            body["filter"] = filter
        if sparse_vector is not None:
            if isinstance(sparse_vector, SparseValues):
                body["sparseVector"] = {
                    "indices": sparse_vector.indices,
                    "values": sparse_vector.values,
                }
            else:
                body["sparseVector"] = sparse_vector
        if scan_factor is not None:
            body["scanFactor"] = scan_factor
        if max_candidates is not None:
            body["maxCandidates"] = max_candidates

        logger.info("Querying index with top_k=%d", top_k)
        response = self._http.post("/query", json=body)
        result = self._adapter.to_query_response(response.content)
        result.response_info = extract_response_info(response)
        logger.debug("Query returned %d matches", len(result.matches))
        return result

    def query_namespaces(
        self,
        *,
        vector: list[float],
        namespaces: list[str],
        metric: str,
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = False,
        sparse_vector: SparseValues | dict[str, Any] | None = None,
        scan_factor: float | None = None,
        max_candidates: int | None = None,
    ) -> QueryNamespacesResults:
        """Query multiple namespaces in parallel and return merged top results.

        Fans out individual ``query()`` calls across all given namespaces
        using a thread pool, then merges results via a heap-based aggregator
        that returns the overall top-k matches ranked by the specified metric.

        Args:
            vector: Dense query vector values (must be non-empty).
            namespaces: Namespaces to query (must be non-empty). Duplicates
                are removed while preserving order.
            metric: Distance metric — ``"cosine"``, ``"euclidean"``, or
                ``"dotproduct"``.
            top_k: Maximum number of results to return. Defaults to 10.
            filter: Metadata filter expression applied to every namespace.
            include_values: Whether to include vector values in results.
            include_metadata: Whether to include metadata in results.
            sparse_vector: Sparse query vector with indices and values.
            scan_factor: DRN performance tuning — controls how much of the
                index is scanned during a query. Higher values scan more
                data and may improve recall at the cost of latency.
            max_candidates: DRN performance tuning — maximum number of
                candidate vectors to consider during the search phase.

        Returns:
            :class:`QueryNamespacesResults` with the merged top-k matches, total
            usage, and per-namespace usage.

        Raises:
            :exc:`ValidationError`: If *namespaces* or *vector* is empty.
            :exc:`ValueError`: If *metric* is not a recognized value.
            :exc:`ApiError`: If any individual namespace query fails.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            results = idx.query_namespaces(
                vector=[0.012, -0.087, 0.153],  # truncated; use your actual dimension
                namespaces=["articles-en", "articles-fr", "articles-de"],
                metric="cosine",
                top_k=10,
            )
            for match in results.matches:
                print(match.id, match.score)
        """
        if not namespaces:
            raise ValidationError("namespaces must be a non-empty list")
        if not vector:
            raise ValidationError("vector must be a non-empty list")

        namespaces = list(dict.fromkeys(namespaces))
        effective_top_k = top_k if top_k is not None else 10
        aggregator = QueryResultsAggregator(metric=metric, top_k=effective_top_k)

        with ThreadPoolExecutor(max_workers=min(len(namespaces), 32)) as pool:
            future_to_ns = {
                pool.submit(
                    self.query,
                    top_k=effective_top_k,
                    vector=vector,
                    namespace=ns,
                    filter=filter,
                    include_values=include_values,
                    include_metadata=include_metadata,
                    sparse_vector=sparse_vector,
                    scan_factor=scan_factor,
                    max_candidates=max_candidates,
                ): ns
                for ns in namespaces
            }
            for future in as_completed(future_to_ns):
                ns = future_to_ns[future]
                response = future.result()
                aggregator.add_results(ns, response)

        return aggregator.get_results()

    def fetch(
        self,
        *,
        ids: list[str],
        namespace: str = "",
    ) -> FetchResponse:
        """Fetch vectors by their IDs from a namespace.

        Args:
            ids (list[str]): List of vector IDs to fetch (must be non-empty).
            namespace (str): Namespace to fetch from. Defaults to the default namespace.

        Returns:
            :class:`FetchResponse` with a map of vector IDs to Vector objects, namespace,
            and usage info. IDs that do not exist are omitted from the map rather
            than raising an error.

        Raises:
            :exc:`ValidationError`: If ids is empty.
            :exc:`ApiError`: If the API returns an error response (e.g. authentication
                failure or server error).
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            response = idx.fetch(ids=["article-101", "article-102"])
            for vid, vec in response.vectors.items():
                print(vid, vec.values)
        """
        if not ids:
            raise ValidationError("ids must be a non-empty list")

        params: dict[str, Any] = {"ids": ids}
        if namespace:
            params["namespace"] = namespace

        logger.info("Fetching %d vectors", len(ids))
        response = self._http.get("/vectors/fetch", params=params)
        result = self._adapter.to_fetch_response(response.content)
        result.response_info = extract_response_info(response)
        logger.debug("Fetched %d vectors", len(result.vectors))
        return result

    def fetch_by_metadata(
        self,
        *,
        filter: dict[str, Any],
        namespace: str = "",
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> FetchByMetadataResponse:
        """Fetch vectors matching a metadata filter expression.

        Returns vectors whose metadata satisfies the given filter, with
        pagination support. The server returns up to 100 vectors per page
        when no limit is specified.

        Args:
            filter: Metadata filter expression (required).
            namespace: Namespace to fetch from. Defaults to the default
                namespace.
            limit: Maximum number of vectors to return per page. When
                ``None``, the server default (100) is used.
            pagination_token: Token from a previous response to fetch the
                next page. When ``None``, fetches the first page.

        Returns:
            :class:`FetchByMetadataResponse` with matched vectors, namespace, usage,
            and pagination token for the next page (if any).

        Raises:
            :exc:`ApiError`: If the API returns an error response (e.g. authentication
                failure or server error).

        Examples:

            response = idx.fetch_by_metadata(
                filter={"genre": {"$eq": "comedy"}},
                namespace="movies",
            )
            for vid, vec in response.vectors.items():
                print(vid, vec.values)

            # Paginate through all results
            token = response.pagination.next if response.pagination else None
            while token:
                response = idx.fetch_by_metadata(
                    filter={"genre": {"$eq": "comedy"}},
                    namespace="movies",
                    pagination_token=token,
                )
                token = response.pagination.next if response.pagination else None
        """
        body: dict[str, Any] = {"filter": filter}
        if namespace:
            body["namespace"] = namespace
        if limit is not None:
            body["limit"] = limit
        if pagination_token is not None:
            body["paginationToken"] = pagination_token

        logger.info("Fetching vectors by metadata")
        response = self._http.post("/vectors/fetch_by_metadata", json=body)
        result = self._adapter.to_fetch_by_metadata_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        delete_all: bool = False,
        filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> None:
        """Delete vectors from a namespace by ID, filter, or delete-all flag.

        Exactly one of ``ids``, ``delete_all``, or ``filter`` must be specified.
        Deleting IDs that do not exist does not raise an error.

        Args:
            ids (list[str] | None): List of vector IDs to delete.
            delete_all (bool): If True, delete all vectors in the namespace.
            filter (dict[str, Any] | None): Metadata filter expression selecting vectors to delete.
            namespace (str): Namespace to delete from. Defaults to the default namespace.

        Returns:
            None — a successful delete returns no payload.

        Raises:
            :exc:`ValidationError`: If zero or more than one deletion mode is specified.
            :exc:`ApiError`: If the API returns an error response (e.g. authentication
                failure or server error).
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            # Delete by IDs
            idx.delete(ids=["article-101", "article-102"])

            # Delete all vectors in a namespace
            idx.delete(delete_all=True, namespace="articles-deprecated")

            # Delete by metadata filter
            idx.delete(filter={"category": {"$eq": "obsolete"}})
        """
        mode_count = sum([ids is not None, delete_all, filter is not None])
        if mode_count == 0:
            raise ValidationError("Must specify one of ids, delete_all, or filter")
        if mode_count > 1:
            raise ValidationError(
                "Cannot combine ids, delete_all, and filter — specify exactly one"
            )

        body: dict[str, Any] = {"namespace": namespace}
        if ids is not None:
            body["ids"] = ids
        if delete_all:
            body["deleteAll"] = True
        if filter is not None:
            body["filter"] = filter

        logger.info("Deleting vectors from namespace %r", namespace)
        self._http.post("/vectors/delete", json=body)

    def update(
        self,
        *,
        id: str | None = None,
        values: list[float] | None = None,
        sparse_values: SparseValues | dict[str, Any] | None = None,
        set_metadata: dict[str, Any] | None = None,
        namespace: str = "",
        filter: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> UpdateResponse:
        """Update vectors by ID or metadata filter.

        Updates a single vector's dense values, sparse values, or metadata by
        identifier, or bulk-updates metadata on all vectors matching a filter.

        Exactly one of ``id`` or ``filter`` must be specified.

        Args:
            id (str | None): ID of the vector to update.
            values (list[float] | None): New dense vector values.
            sparse_values (SparseValues | dict[str, Any] | None): New sparse vector with ``indices``
                and ``values`` keys.
            set_metadata (dict[str, Any] | None): Metadata fields to set or overwrite.
            namespace (str): Namespace to target. Defaults to the default namespace.
            filter (dict[str, Any] | None): Metadata filter expression selecting vectors to update.
            dry_run (bool): If True, return the count of records that would be
                affected without applying changes. Only applies to filter-based
                updates.

        Returns:
            :class:`UpdateResponse` with matched_records count (when available).

        Raises:
            :exc:`ValidationError`: If both or neither of id and filter are provided.
            :exc:`ApiError`: If the API returns an error response (e.g. authentication
                failure or server error).
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            # Update by ID
            # truncated; use your actual dimension
            idx.update(id="article-101", values=[0.012, -0.087, 0.153])

            # Bulk-update metadata by filter
            idx.update(
                filter={"genre": {"$eq": "drama"}},
                set_metadata={"year": 2020},
            )
        """
        has_id = id is not None
        has_filter = filter is not None
        if has_id and has_filter:
            raise ValidationError("Exactly one of id or filter must be provided, not both")
        if not has_id and not has_filter:
            raise ValidationError("Exactly one of id or filter must be provided, got neither")

        body: dict[str, Any] = {"namespace": namespace}
        if id is not None:
            body["id"] = id
        if values is not None:
            body["values"] = values
        if sparse_values is not None:
            if isinstance(sparse_values, SparseValues):
                body["sparseValues"] = {
                    "indices": sparse_values.indices,
                    "values": sparse_values.values,
                }
            else:
                body["sparseValues"] = sparse_values
        if set_metadata is not None:
            body["setMetadata"] = set_metadata
        if filter is not None:
            body["filter"] = filter
        if dry_run:
            body["dryRun"] = True

        logger.info("Updating vectors in namespace %r", namespace)
        response = self._http.post("/vectors/update", json=body)
        result = self._adapter.to_update_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    def describe_index_stats(
        self,
        *,
        filter: dict[str, Any] | None = None,
    ) -> DescribeIndexStatsResponse:
        """Return statistics for this index.

        Returns aggregate statistics including total vector count,
        per-namespace vector counts, dimension, and index fullness.

        Args:
            filter (dict[str, Any] | None): Metadata filter expression. When
                provided, only vectors matching the filter are counted.

        Returns:
            :class:`DescribeIndexStatsResponse` with namespace summaries, dimension,
            total vector count, and fullness metrics.

        Raises:
            :exc:`ApiError`: If the API returns an error response (e.g. authentication
                failure or server error).
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            stats = idx.describe_index_stats()
            print(stats.total_vector_count, stats.dimension)

            # With filter — only count vectors matching the expression
            stats = idx.describe_index_stats(
                filter={"genre": {"$eq": "drama"}}
            )
        """
        body: dict[str, Any] = {}
        if filter is not None:
            body["filter"] = filter

        logger.info("Describing index stats")
        response = self._http.post("/describe_index_stats", json=body)
        result = self._adapter.to_stats_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    def search(
        self,
        *,
        namespace: str,
        top_k: int,
        inputs: dict[str, Any] | None = None,
        vector: list[float] | None = None,
        id: str | None = None,
        filter: dict[str, Any] | None = None,
        fields: list[str] | None = None,
        rerank: RerankConfig | dict[str, Any] | None = None,
        match_terms: dict[str, Any] | None = None,
    ) -> SearchRecordsResponse:
        """Search records by text, vector, or ID with optional reranking.

        Searches a namespace using integrated inference (text inputs embedded
        server-side), a raw vector, or an existing record ID as the query.

        .. note::
           Use this method for indexes with integrated inference. For classic
           indexes where you provide your own vectors, use :meth:`query`.

        Args:
            namespace (str): Namespace to search in (required).
            top_k (int): Number of results to return (must be >= 1).
            inputs (dict[str, Any] | None): Inputs for server-side embedding
                (e.g. ``{"text": "query text"}``).
            vector (list[float] | None): Dense query vector values.
            id (str | None): ID of an existing record to use as the query.
            filter (dict[str, Any] | None): Metadata filter expression.
            fields (list[str] | None): Field names to include in results.
                When ``None``, the server returns all available fields.
            rerank (dict[str, Any] | None): Reranking configuration with
                ``model`` (required), ``rank_fields`` (required), and optional
                ``top_n``, ``parameters``, ``query`` keys.
            match_terms (dict[str, Any] | None): Term-matching constraint for
                sparse search. Requires keys ``"strategy"`` (currently only
                ``"all"``) and ``"terms"`` (list of strings). Only supported
                for sparse indexes using ``pinecone-sparse-english-v0``.
                ``None`` disables term matching.

        Returns:
            :class:`SearchRecordsResponse` with hits and usage statistics.

        Raises:
            :exc:`ValidationError`: If ``namespace`` is not a string, ``top_k < 1``,
                or ``rerank`` is missing required keys.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            response = idx.search(
                namespace="articles-en",
                top_k=10,
                inputs={"text": "benefits of vector databases for search"},
            )
            for hit in response.result.hits:
                print(hit.id, hit.score)

            Search with reranking:

                response = idx.search(
                    namespace="articles-en",
                    top_k=10,
                    inputs={"text": "benefits of vector databases"},
                    rerank={
                        "model": "bge-reranker-v2-m3",
                        "rank_fields": ["text"],
                        "top_n": 5,
                    },
                )
                for hit in response.result.hits:
                    print(hit.id, hit.score)

        .. note::
           Use inline ``rerank`` when searching and reranking in a single call.
           Use ``pc.inference.rerank()`` when reranking results from a different
           source or when you need to rerank without searching.
        """
        if not isinstance(namespace, str):
            raise ValidationError("namespace must be a string")
        if top_k < 1:
            raise ValidationError(f"top_k must be a positive integer, got {top_k}")
        if rerank is not None:
            if "model" not in rerank:
                raise ValidationError("rerank requires 'model' to be specified")
            if "rank_fields" not in rerank:
                raise ValidationError("rerank requires 'rank_fields' to be specified")
        if inputs is None and vector is None and id is None:
            raise ValidationError(
                "At least one of inputs, vector, or id must be provided as a query source"
            )

        query_body: dict[str, Any] = {"top_k": top_k}
        if inputs is not None:
            query_body["inputs"] = inputs
        if vector is not None:
            query_body["vector"] = vector
        if id is not None:
            query_body["id"] = id
        if filter is not None:
            query_body["filter"] = filter
        if match_terms is not None:
            query_body["match_terms"] = match_terms

        body: dict[str, Any] = {"query": query_body}
        if fields is not None:
            body["fields"] = fields
        if rerank is not None:
            body["rerank"] = rerank

        logger.info("Searching namespace %r with top_k=%d", namespace, top_k)
        response = self._http.post(f"/records/namespaces/{namespace}/search", json=body)
        result = self._adapter.to_search_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    def search_records(
        self,
        *,
        namespace: str,
        top_k: int,
        inputs: dict[str, Any] | None = None,
        vector: list[float] | None = None,
        id: str | None = None,
        filter: dict[str, Any] | None = None,
        fields: list[str] | None = None,
        rerank: dict[str, Any] | None = None,
        match_terms: dict[str, Any] | None = None,
    ) -> SearchRecordsResponse:
        """Alias for :meth:`search`.

        Prefer calling :meth:`search` directly — this alias exists for backwards compatibility.
        """
        return self.search(
            namespace=namespace,
            top_k=top_k,
            inputs=inputs,
            vector=vector,
            id=id,
            filter=filter,
            fields=fields,
            rerank=rerank,
            match_terms=match_terms,
        )

    def create_namespace(
        self,
        *,
        name: str,
        schema: dict[str, Any] | None = None,
    ) -> NamespaceDescription:
        """Create a named namespace in the index.

        Args:
            name (str): Name for the new namespace (must be non-empty).
            schema (dict[str, Any] | None): Optional schema configuration
                with metadata field indexing settings.

        Returns:
            :class:`NamespaceDescription` with the namespace name and record count.

        Raises:
            :exc:`ValidationError`: If the name is not a string or is empty/whitespace.
            :exc:`ApiError`: If the API returns an error response (e.g. 409 conflict
                when namespace already exists).
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            ns = idx.create_namespace(name="movies-en")
            print(ns.name, ns.record_count)

            ns = idx.create_namespace(
                name="movies-en",
                schema={"fields": {"genre": {"filterable": True}}},
            )
        """
        if not isinstance(name, str):
            raise ValidationError("namespace name must be a string")
        if not name or not name.strip():
            raise ValidationError("namespace name must be a non-empty string")

        body: dict[str, Any] = {"name": name}
        if schema is not None:
            body["schema"] = schema

        logger.info("Creating namespace %r", name)
        response = self._http.post("/namespaces", json=body)
        return self._adapter.to_namespace_description(response.content)

    def describe_namespace(
        self,
        *,
        name: str,
    ) -> NamespaceDescription:
        """Describe a namespace by name.

        Args:
            name (str): Name of the namespace to describe.

        Returns:
            :class:`NamespaceDescription` with the namespace name, record count,
            and schema information.

        Raises:
            :exc:`ValidationError`: If the name is not a string or is empty/whitespace.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            ns = idx.describe_namespace(name="movies-en")
            print(ns.name, ns.record_count)
        """
        if not isinstance(name, str):
            raise ValidationError("namespace name must be a string")
        if not name or not name.strip():
            raise ValidationError("namespace name must be a non-empty string")

        logger.info("Describing namespace %r", name)
        response = self._http.get(f"/namespaces/{name}")
        return self._adapter.to_namespace_description(response.content)

    def delete_namespace(
        self,
        *,
        name: str,
    ) -> None:
        """Delete a namespace by name, removing all its vectors.

        Args:
            name (str): Name of the namespace to delete.

        Returns:
            None — a successful delete returns no payload.

        Raises:
            :exc:`ValidationError`: If the name is not a string or is empty/whitespace.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            idx.delete_namespace(name="movies-deprecated")
        """
        if not isinstance(name, str):
            raise ValidationError("namespace name must be a string")
        if not name or not name.strip():
            raise ValidationError("namespace name must be a non-empty string")

        logger.info("Deleting namespace %r", name)
        self._http.delete(f"/namespaces/{name}")

    def list_namespaces_paginated(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> ListNamespacesResponse:
        """Fetch a single page of namespace descriptions.

        Args:
            prefix (str | None): Return only namespaces whose names start with this prefix.
            limit (int | None): Maximum number of namespaces to return in this page.
            pagination_token (str | None): Token from a previous response to fetch the next page.

        Returns:
            :class:`ListNamespacesResponse` with namespace descriptions, pagination info,
            and total count.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            response = idx.list_namespaces_paginated(prefix="prod-", limit=10)
            for ns in response.namespaces:
                print(ns.name, ns.record_count)
        """
        params: dict[str, Any] = {}
        if prefix is not None:
            params["prefix"] = prefix
        if limit is not None:
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        logger.info("Listing namespaces")
        response = self._http.get("/namespaces", params=params)
        return self._adapter.to_list_namespaces_response(response.content)

    def list_namespaces(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
    ) -> Iterator[ListNamespacesResponse]:
        """List namespaces, automatically following pagination.

        Yields one ``ListNamespacesResponse`` per page. The generator
        automatically follows pagination tokens until all pages have been
        retrieved.

        Args:
            prefix (str | None): Return only namespaces whose names start with this prefix.
            limit (int | None): Maximum number of namespaces to return per page.

        Yields:
            :class:`ListNamespacesResponse` for each page of results.

        Examples:

            for page in idx.list_namespaces(prefix="prod-"):
                for ns in page.namespaces:
                    print(ns.name, ns.record_count)
        """
        pagination_token: str | None = None
        while True:
            page = self.list_namespaces_paginated(
                prefix=prefix,
                limit=limit,
                pagination_token=pagination_token,
            )
            if page.namespaces:
                yield page
            if page.pagination is not None and page.pagination.next is not None:
                pagination_token = page.pagination.next
            else:
                break

    def list_paginated(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        namespace: str = "",
    ) -> ListResponse:
        """Fetch a single page of vector IDs from a namespace.

        Args:
            prefix (str | None): Return only IDs starting with this prefix.
            limit (int | None): Maximum number of IDs to return in this page.
            pagination_token (str | None): Token from a previous response to fetch the next page.
            namespace (str): Namespace to list from. Defaults to the default namespace.

        Returns:
            :class:`ListResponse` with vector IDs, pagination info, namespace, and usage.

        Raises:
            :exc:`ValidationError`: If inputs are invalid.
            :exc:`ApiError`: If the API returns an error response (e.g. authentication
                failure or server error).

        Examples:

            response = idx.list_paginated(prefix="doc1#", limit=50)
            for item in response.vectors:
                print(item.id)
        """
        params: dict[str, Any] = {"namespace": namespace}
        if prefix is not None:
            params["prefix"] = prefix
        if limit is not None:
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        logger.info("Listing vectors in namespace %r", namespace)
        response = self._http.get("/vectors/list", params=params)
        result = self._adapter.to_list_response(response.content)
        result.response_info = extract_response_info(response)
        return result

    def list(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        namespace: str = "",
    ) -> Iterator[ListResponse]:
        """List vector IDs in a namespace, automatically following pagination.

        Yields one ``ListResponse`` per page. The generator automatically
        follows pagination tokens until all pages have been retrieved.

        Args:
            prefix (str | None): Return only IDs starting with this prefix.
            limit (int | None): Maximum number of IDs to return per page.
            namespace (str): Namespace to list from. Defaults to the default namespace.

        Yields:
            :class:`ListResponse` for each page of results.

        Examples:

            for page in idx.list(prefix="doc1#"):
                for item in page.vectors:
                    print(item.id)
        """
        pagination_token: str | None = None
        while True:
            page = self.list_paginated(
                prefix=prefix,
                limit=limit,
                pagination_token=pagination_token,
                namespace=namespace,
            )
            if page.vectors:
                yield page
            if page.pagination is not None and page.pagination.next is not None:
                pagination_token = page.pagination.next
            else:
                break

    def _validate_import_id(self, id: str | int) -> str:
        """Validate and normalize an import operation ID.

        Args:
            id: Import operation ID. If int, converted to str silently.

        Returns:
            The validated string ID.

        Raises:
            :exc:`ValidationError`: If the ID is empty or exceeds 1000 characters.
        """
        str_id = str(id) if isinstance(id, int) else id
        if not str_id or len(str_id) > 1000:
            raise ValidationError(
                "import id must be between 1 and 1000 characters, "
                f"got {len(str_id) if str_id else 0}"
            )
        return str_id

    def start_import(
        self,
        uri: str,
        *,
        error_mode: str = "continue",
        integration_id: str | None = None,
    ) -> StartImportResponse:
        """Start a bulk import operation from an external data source.

        Initiates an asynchronous bulk import of vectors from cloud storage
        into the index. The import runs server-side; use :meth:`describe_import`
        to poll for progress and completion.

        .. note::
           The import URI must point to a directory of Parquet files in cloud
           storage (``s3://`` or ``gs://``). Each Parquet file must follow the
           Pinecone-required schema. See
           `Pinecone import docs <https://docs.pinecone.io/guides/data/understanding-imports>`_
           for the required Parquet schema and supported storage formats.

        Args:
            uri (str): Source URI for the import data (e.g.
                ``"s3://my-bucket/vectors/"`` or ``"gs://my-bucket/vectors/"``).
            error_mode (str): How to handle errors during import. Must be
                ``"continue"`` (default) or ``"abort"``. Case-insensitive.
            integration_id (str | None): Optional integration ID for the import.

        Returns:
            :class:`StartImportResponse` with the ID of the created import
            operation.

        Raises:
            :exc:`ValidationError`: If ``error_mode`` is not ``"continue"`` or ``"abort"``.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:
            Start an import and poll until complete:

            >>> import time
            >>> response = idx.start_import(uri="s3://my-bucket/vectors/")
            >>> import_id = response.id
            >>>
            >>> # Poll until the import finishes
            >>> import_op = idx.describe_import(import_id)
            >>> while import_op.status not in ("Completed", "Failed", "Cancelled"):
            ...     time.sleep(10)
            ...     import_op = idx.describe_import(import_id)
            >>> print(f"Status: {import_op.status}, records imported: {import_op.records_imported}")

            Abort on first error instead of continuing:

            >>> response = idx.start_import(
            ...     uri="s3://my-bucket/vectors/",
            ...     error_mode="abort",
            ... )

        .. seealso::
           - :meth:`upsert` — for upserting vectors directly in small
             batches (single request per call).
           - :meth:`upsert_records` — for indexes with integrated inference
             (text in, server-side embedding).
           - :meth:`upsert_from_dataframe` — for loading vectors from a
             pandas DataFrame with automatic batching.
        """
        error_mode = error_mode.lower()
        if error_mode not in ("continue", "abort"):
            raise ValidationError(f"error_mode must be 'continue' or 'abort', got {error_mode!r}")

        body: dict[str, Any] = {
            "uri": uri,
            "errorMode": {"onError": error_mode},
        }
        if integration_id is not None:
            body["integrationId"] = integration_id

        logger.info("Starting bulk import from %s", uri)
        response = self._http.post("/bulk/imports", json=body)
        return self._imports_adapter.to_start_import_response(response.content)

    def describe_import(self, id: str | int) -> ImportModel:
        """Describe a bulk import operation by ID.

        Args:
            id: Import operation ID. Integers are converted to strings silently.

        Returns:
            :class:`ImportModel` with the import operation details.

        Raises:
            :exc:`ValidationError`: If the ID is empty or exceeds 1000 characters.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            import_op = idx.describe_import("import-123")
            print(import_op.status, import_op.percent_complete)
        """
        str_id = self._validate_import_id(id)
        logger.info("Describing import %s", str_id)
        response = self._http.get(f"/bulk/imports/{str_id}")
        return self._imports_adapter.to_import_model(response.content)

    def cancel_import(self, id: str | int) -> None:
        """Cancel a bulk import operation by ID.

        Args:
            id: Import operation ID. Integers are converted to strings silently.

        Returns:
            None — a successful cancellation returns no payload.

        Raises:
            :exc:`ValidationError`: If the ID is empty or exceeds 1000 characters.
            :exc:`ApiError`: If the API returns an error response.
            :exc:`PineconeConnectionError`: If a network-level connection
                fails (DNS, refused, transport error).
            :exc:`PineconeTimeoutError`: If the request exceeds the configured timeout.

        Examples:

            idx.cancel_import("import-123")
        """
        str_id = self._validate_import_id(id)
        logger.info("Cancelling import %s", str_id)
        self._http.delete(f"/bulk/imports/{str_id}")

    def list_imports(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Iterator[ImportModel]:
        """List bulk import operations, automatically following pagination.

        Yields individual :class:`ImportModel` objects, fetching additional
        pages transparently until all results have been returned.

        Args:
            limit (int | None): Maximum number of imports per page
                (max 100, server default 100).
            pagination_token (str | None): Token to resume pagination
                from a previous call.

        Yields:
            :class:`ImportModel` for each import operation.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            for imp in idx.list_imports():
                print(imp.id, imp.status)
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        while True:
            response = self._http.get("/bulk/imports", params=params)
            import_list = self._imports_adapter.to_import_list(response.content)
            yield from import_list
            next_token = import_list.pagination.next if import_list.pagination else None
            if next_token is None:
                break
            params["paginationToken"] = next_token

    def list_imports_paginated(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> ImportList:
        """Fetch a single page of bulk import operations.

        Returns an :class:`ImportList` for one page. The caller is responsible
        for managing the pagination token.

        Args:
            limit (int | None): Maximum number of imports to return in this page.
            pagination_token (str | None): Token from a previous response to
                fetch the next page.

        Returns:
            :class:`ImportList` with the import operations for the requested page.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:

            page = idx.list_imports_paginated(limit=10)
            for imp in page:
                print(imp.id, imp.status)
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        response = self._http.get("/bulk/imports", params=params)
        return self._imports_adapter.to_import_list(response.content)

    def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        self._http.close()

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return f"Index(host='{self._host}')"
