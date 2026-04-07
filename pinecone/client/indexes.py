"""Indexes namespace — list, describe, create, and exists operations."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.indexes_adapter import IndexesAdapter
from pinecone._internal.indexes_helpers import (
    build_byoc_body,
    build_create_body,
    build_integrated_body,
    poll_index_until_ready,
    resolve_enum_value,
    validate_byoc_inputs,
    validate_create_inputs,
    validate_integrated_inputs,
    validate_read_capacity,
)
from pinecone._internal.validation import require_non_empty
from pinecone.errors.exceptions import (
    NotFoundError,
    PineconeTimeoutError,
    ValidationError,
)
from pinecone.models.enums import DeletionProtection, Metric, VectorType
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.list import IndexList
from pinecone.models.indexes.specs import ByocSpec, IntegratedSpec, PodSpec, ServerlessSpec

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 5


class Indexes:
    """Control-plane operations for Pinecone indexes.

    Provides ``list``, ``describe``, ``exists``, ``create``, ``delete``,
    and ``configure`` methods.

    .. seealso::
       Use :meth:`Pinecone.index(name) <pinecone.Pinecone.index>` to get a
       data-plane client for vector operations on a specific index.

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Examples:

        from pinecone import Pinecone

        pc = Pinecone(api_key="your-api-key")
        for idx in pc.indexes.list():
            print(idx.name)
    """

    def __init__(self, http: HTTPClient, host_cache: dict[str, str] | None = None) -> None:
        self._http = http
        self._adapter = IndexesAdapter()
        self._host_cache: dict[str, str] = host_cache if host_cache is not None else {}

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Indexes()"

    def list(self) -> IndexList:
        """List all indexes in the project.

        Returns all indexes in a single response without filtering,
        sorting, or pagination.

        Returns:
            :class:`IndexList` supporting iteration, len(), index access,
            and a names() convenience method.

        Raises:
            :exc:`ApiError`: If the API returns an error response (e.g. authentication
                failure or server error).

        Examples:

            indexes = pc.indexes.list()
            print(indexes.names())
            for idx in indexes:
                print(idx.name, idx.metric)
        """
        logger.info("Listing indexes")
        response = self._http.get("/indexes")
        result = self._adapter.to_index_list(response.content)
        logger.debug("Listed %d indexes", len(result))
        return result

    def describe(self, name: str) -> IndexModel:
        """Get detailed information about a named index.

        After a successful call the host URL is cached internally for
        later data-plane client construction.

        Args:
            name (str): The name of the index to describe.

        Returns:
            :class:`IndexModel` with name, dimension, metric, host, spec,
            status, deletion_protection, vector_type, and tags.

        Raises:
            :exc:`ValidationError`: If *name* is empty.
            :exc:`NotFoundError`: If the index does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:

            desc = pc.indexes.describe("my-index")
            print(desc.host)
        """
        require_non_empty("name", name)
        logger.info("Describing index %r", name)
        response = self._http.get(f"/indexes/{name}")
        model = self._adapter.to_index_model(response.content)
        self._host_cache[name] = model.host
        logger.debug("Described index %r (host=%s)", name, model.host)
        return model

    def exists(self, name: str) -> bool:
        """Check whether a named index exists.

        Uses describe internally; returns ``True`` on success and
        ``False`` when a 404 is returned.

        Args:
            name (str): The name of the index to check.

        Returns:
            True if the index exists, False otherwise.

        Raises:
            :exc:`ValidationError`: If *name* is empty.
            :exc:`ApiError`: If the API returns an error other than 404.

        Examples:

            if pc.indexes.exists("my-index"):
                print("Index found")
        """
        require_non_empty("name", name)
        try:
            self.describe(name)
            return True
        except NotFoundError:
            return False

    def delete(self, name: str, *, timeout: int | None = None) -> None:
        """Delete an index by name.

        After sending the delete request, removes the cached host URL
        for the index. By default, polls every 5 seconds until the index
        disappears with no upper time bound.

        Args:
            name (str): The name of the index to delete.
            timeout (int | None): Seconds to wait for the index to disappear.
                Use ``None`` (default) to poll indefinitely until the index
                is gone. Use a positive int to poll with a deadline.
                Use ``-1`` to return immediately without polling.

        Raises:
            :exc:`ValidationError`: If *name* is empty.
            :exc:`NotFoundError`: If the index does not exist.
            :exc:`PineconeTimeoutError`: If the index still exists after *timeout* seconds.
            :exc:`ApiError`: If the API returns another error response.

        Examples:

            pc.indexes.delete("my-index")

            # Wait up to 60 seconds for deletion to complete
            pc.indexes.delete("my-index", timeout=60)
        """
        require_non_empty("name", name)
        logger.info("Deleting index %r", name)
        self._http.delete(f"/indexes/{name}")
        self._host_cache.pop(name, None)
        logger.debug("Deleted index %r", name)

        if timeout == -1:
            return

        start = time.monotonic()
        while True:
            try:
                self.describe(name)
            except NotFoundError:
                return
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise PineconeTimeoutError(f"Index '{name}' still exists after {timeout}s")
            time.sleep(_POLL_INTERVAL_SECONDS)

    def configure(
        self,
        name: str,
        *,
        replicas: int | None = None,
        pod_type: str | None = None,
        deletion_protection: DeletionProtection | str | None = None,
        tags: dict[str, str] | None = None,
        read_capacity: dict[str, Any] | None = None,
    ) -> None:
        """Configure an existing index.

        Updates mutable properties of an index such as replicas, pod type,
        deletion protection, tags, and read capacity.

        Args:
            name (str): The name of the index to configure.
            replicas (int | None): Number of replicas for pod-based indexes.
            pod_type (str | None): Pod type for pod-based indexes (e.g. ``"p1.x2"``).
            deletion_protection (DeletionProtection | str | None): ``"enabled"`` or ``"disabled"``.
            tags (dict[str, str] | None): Key-value tags to merge with existing tags.
                Set a value to ``""`` to remove a tag.
            read_capacity (dict[str, Any] | None): Read capacity configuration for
                BYOC indexes. Pass ``{"mode": "OnDemand"}`` or
                ``{"mode": "Dedicated", "dedicated": {"node_type": "t1",
                "scaling": "Manual", "manual": {"replicas": 2, "shards": 1}}}``.

        Raises:
            :exc:`ValidationError`: If *name* is empty or *read_capacity* is invalid.
            :exc:`NotFoundError`: If the index does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:

            pc.indexes.configure("my-index", replicas=4)
            pc.indexes.configure("my-index", tags={"env": "prod"})
        """
        require_non_empty("name", name)
        logger.info("Configuring index %r", name)

        body: dict[str, Any] = {}

        # Pod spec fields
        pod_fields: dict[str, Any] = {}
        if replicas is not None:
            pod_fields["replicas"] = replicas
        if pod_type is not None:
            pod_fields["pod_type"] = pod_type

        # BYOC read capacity — mutually exclusive with pod fields
        if pod_fields and read_capacity is not None:
            raise ValidationError(
                "Cannot specify both pod fields (replicas, pod_type) and"
                " read_capacity in the same configure call — they apply"
                " to different index types"
            )

        if pod_fields:
            body["spec"] = {"pod": pod_fields}

        if read_capacity is not None:
            validate_read_capacity(read_capacity)
            body["spec"] = {"byoc": {"read_capacity": read_capacity}}

        # Deletion protection — only include when explicitly specified
        if deletion_protection is not None:
            body["deletion_protection"] = resolve_enum_value(deletion_protection)

        # Tag merging — fetch current tags and merge
        if tags is not None:
            current = self.describe(name)
            merged = {**(current.tags or {}), **tags}
            body["tags"] = merged

        self._http.patch(f"/indexes/{name}", json=body)
        logger.debug("Configured index %r", name)

    def create(
        self,
        *,
        name: str,
        spec: ServerlessSpec | PodSpec | ByocSpec | IntegratedSpec | dict[str, Any],
        dimension: int | None = None,
        metric: Metric | str = "cosine",
        vector_type: VectorType | str = "dense",
        deletion_protection: DeletionProtection | str = "disabled",
        tags: dict[str, str] | None = None,
        schema: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> IndexModel:
        """Create a new Pinecone index.

        Supports serverless, pod-based, BYOC (bring your own cloud), and
        integrated (model-backed) index creation. Integrated indexes use
        Pinecone's built-in embedding models so dimension and metric are
        inferred from the model.

        Args:
            name (str): Name for the new index.
            spec (ServerlessSpec | PodSpec | ByocSpec | IntegratedSpec | dict[str, Any]):
                Deployment spec — a ServerlessSpec, PodSpec, ByocSpec,
                IntegratedSpec, or raw dict.
            dimension (int | None): Vector dimension (required for dense
                non-integrated indexes).
            metric (Metric | str): Similarity metric (cosine, euclidean, dotproduct).
            vector_type (VectorType | str): Vector type (dense or sparse).
            deletion_protection (DeletionProtection | str): Whether deletion protection is enabled.
            tags (dict[str, str] | None): Optional key-value tags.
            schema (dict[str, Any] | None): Optional metadata schema defining
                field types for indexing. Accepts both flat format
                (``{"field": {"type": "str"}}``) and nested format
                (``{"fields": {"field": {"type": "str"}}}``).
            timeout (int | None): Seconds to wait for the index to become ready.
                Use ``None`` (default) to poll indefinitely every 5 seconds
                with no upper time bound. Use a positive int to poll with a
                deadline. Use ``-1`` to return immediately without polling.
                Raises ``PineconeTimeoutError`` if the index is not ready
                before the deadline. ``IndexInitFailedError`` if
                initialization fails.

        Returns:
            :class:`IndexModel` describing the created index.

        Raises:
            :exc:`ValidationError`: If inputs fail client-side validation.
            :exc:`NotFoundError`: If the index disappears during readiness polling.
            :exc:`IndexInitFailedError`: If the index fails to initialise.
            :exc:`PineconeTimeoutError`: If the index is not ready before the deadline.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            Create a serverless index for storing embeddings:

            >>> from pinecone import Pinecone, ServerlessSpec
            >>> pc = Pinecone(api_key="your-api-key")
            >>> pc.indexes.create(
            ...     name="movie-recommendations",
            ...     dimension=1536,
            ...     spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            ... )

            Create an integrated index with a built-in embedding model:

            >>> from pinecone import Pinecone, IntegratedSpec, EmbedConfig
            >>> pc = Pinecone(api_key="your-api-key")
            >>> pc.indexes.create(
            ...     name="semantic-search",
            ...     spec=IntegratedSpec(
            ...         cloud="aws",
            ...         region="us-east-1",
            ...         embed=EmbedConfig(
            ...             model="multilingual-e5-large",
            ...             field_map={"text": "my_text_field"},
            ...         ),
            ...     ),
            ... )
        """
        if isinstance(spec, IntegratedSpec):
            validate_integrated_inputs(name=name, spec=spec)
            body = build_integrated_body(
                name=name,
                spec=spec,
                deletion_protection=deletion_protection,
                tags=tags,
            )
        elif isinstance(spec, ByocSpec):
            validate_byoc_inputs(
                name=name,
                spec=spec,
                dimension=dimension,
                deletion_protection=deletion_protection,
            )
            body = build_byoc_body(
                name=name,
                spec=spec,
                dimension=dimension,
                metric=metric,
                vector_type=vector_type,
                deletion_protection=deletion_protection,
                tags=tags,
            )
        else:
            validate_create_inputs(
                name=name,
                spec=spec,
                dimension=dimension,
                metric=metric,
                vector_type=vector_type,
                deletion_protection=deletion_protection,
            )
            body = build_create_body(
                name=name,
                spec=spec,
                dimension=dimension,
                metric=metric,
                vector_type=vector_type,
                deletion_protection=deletion_protection,
                tags=tags,
                schema=schema,
            )

        logger.info("Creating index %r", name)
        response = self._http.post("/indexes", json=body)
        model = self._adapter.to_index_model(response.content)
        logger.debug("Created index %r", name)

        if timeout != -1:
            model = self._poll_until_ready(name, timeout)

        return model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _poll_until_ready(self, name: str, timeout: int | None) -> IndexModel:
        """Poll describe() until the index is ready or timeout is reached."""
        return poll_index_until_ready(self.describe, name, timeout)
