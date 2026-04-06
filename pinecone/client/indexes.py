"""Indexes namespace — list, describe, create, and exists operations."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import msgspec

from pinecone._internal.adapters.indexes_adapter import IndexesAdapter
from pinecone._internal.validation import require_non_empty
from pinecone.errors.exceptions import (
    IndexInitFailedError,
    NotFoundError,
    PineconeTimeoutError,
    ValidationError,
)
from pinecone.models.enums import DeletionProtection, Metric, VectorType
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.list import IndexList
from pinecone.models.indexes.specs import IntegratedSpec, PodSpec, ServerlessSpec

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)

_VALID_METRICS = frozenset({"cosine", "euclidean", "dotproduct"})
_VALID_DELETION_PROTECTION = frozenset({"enabled", "disabled"})
_POLL_INTERVAL_SECONDS = 5


class Indexes:
    """Control-plane operations for Pinecone indexes.

    Provides methods to list, describe, and check existence of indexes.

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Examples:

        from pinecone import Pinecone

        pc = Pinecone(api_key="your-api-key")
        for idx in pc.indexes.list():
            print(idx.name)
    """

    def __init__(self, http: HTTPClient) -> None:
        self._http = http
        self._adapter = IndexesAdapter()
        self._host_cache: dict[str, str] = {}

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Indexes()"

    def list(self) -> IndexList:
        """List all indexes in the project.

        Returns all indexes in a single response without filtering,
        sorting, or pagination.

        Returns:
            An IndexList supporting iteration, len(), index access,
            and a names() convenience method.

        Raises:
            ApiError: If the API returns an error response (e.g. authentication
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
            An IndexModel with name, dimension, metric, host, spec,
            status, deletion_protection, vector_type, and tags.

        Raises:
            ValidationError: If *name* is empty.
            NotFoundError: If the index does not exist.
            ApiError: If the API returns another error response.

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
            ValidationError: If *name* is empty.
            ApiError: If the API returns an error other than 404.

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
        for the index. By default, returns immediately without polling.

        Args:
            name (str): The name of the index to delete.
            timeout (int | None): Seconds to wait for the index to disappear.
                Use ``None`` (default) to return immediately without polling.
                Use a positive int to poll until the index is gone or the
                deadline is reached.

        Raises:
            ValidationError: If *name* is empty.
            NotFoundError: If the index does not exist.
            PineconeError: If the index still exists after *timeout* seconds.
            ApiError: If the API returns another error response.

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

        if timeout is None:
            return

        start = time.monotonic()
        while True:
            try:
                self.describe(name)
            except NotFoundError:
                return
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
    ) -> None:
        """Configure an existing index.

        Updates mutable properties of an index such as replicas, pod type,
        deletion protection, and tags.

        Args:
            name (str): The name of the index to configure.
            replicas (int | None): Number of replicas for pod-based indexes.
            pod_type (str | None): Pod type for pod-based indexes (e.g. ``"p1.x2"``).
            deletion_protection (DeletionProtection | str | None): ``"enabled"`` or ``"disabled"``.
            tags (dict[str, str] | None): Key-value tags to merge with existing tags.
                Set a value to ``""`` to remove a tag.

        Raises:
            ValidationError: If *name* is empty.
            NotFoundError: If the index does not exist.
            ApiError: If the API returns another error response.

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
        if pod_fields:
            body["spec"] = {"pod": pod_fields}

        # Deletion protection — only include when explicitly specified
        if deletion_protection is not None:
            body["deletion_protection"] = self._resolve_value(deletion_protection)

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
        spec: ServerlessSpec | PodSpec | IntegratedSpec | dict[str, Any],
        dimension: int | None = None,
        metric: Metric | str = "cosine",
        vector_type: VectorType | str = "dense",
        deletion_protection: DeletionProtection | str = "disabled",
        tags: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> IndexModel:
        """Create a new Pinecone index.

        Supports serverless, pod-based, and integrated (model-backed) index
        creation. Integrated indexes use Pinecone's built-in embedding models
        so dimension and metric are inferred from the model.

        Args:
            name (str): Name for the new index.
            spec (ServerlessSpec | PodSpec | IntegratedSpec | dict[str, Any]):
                Deployment spec — a ServerlessSpec, PodSpec, IntegratedSpec,
                or raw dict.
            dimension (int | None): Vector dimension (required for dense
                non-integrated indexes).
            metric (Metric | str): Similarity metric (cosine, euclidean, dotproduct).
            vector_type (VectorType | str): Vector type (dense or sparse).
            deletion_protection (DeletionProtection | str): Whether deletion protection is enabled.
            tags (dict[str, str] | None): Optional key-value tags.
            timeout (int | None): Seconds to wait for the index to become ready.
                Use ``None`` (default) or ``-1`` to return immediately
                without polling. Use a positive int to poll until the
                index is ready or the deadline is reached. Raises
                ``PineconeError`` if the index is not ready before the
                deadline or if initialization fails.

        Returns:
            An IndexModel describing the created index.

        Raises:
            ValidationError: If inputs fail client-side validation.
            NotFoundError: If the index disappears during readiness polling.
            PineconeError: If the index fails to initialise or times out.
            ApiError: If the API returns another error response.

        Examples:

            pc.indexes.create(
                name="my-index",
                dimension=1536,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

            pc.indexes.create(
                name="my-integrated-index",
                spec=IntegratedSpec(
                    cloud="aws",
                    region="us-east-1",
                    embed=EmbedConfig(
                        model="multilingual-e5-large",
                        field_map={"text": "my_text_field"},
                    ),
                ),
            )
        """
        if isinstance(spec, IntegratedSpec):
            self._validate_integrated_inputs(name=name, spec=spec)
            body = self._build_integrated_body(
                name=name,
                spec=spec,
                deletion_protection=deletion_protection,
                tags=tags,
            )
        else:
            self._validate_create_inputs(
                name=name,
                spec=spec,
                dimension=dimension,
                metric=metric,
                vector_type=vector_type,
                deletion_protection=deletion_protection,
            )
            body = self._build_create_body(
                name=name,
                spec=spec,
                dimension=dimension,
                metric=metric,
                vector_type=vector_type,
                deletion_protection=deletion_protection,
                tags=tags,
            )

        logger.info("Creating index %r", name)
        response = self._http.post("/indexes", json=body)
        model = self._adapter.to_index_model(response.content)
        logger.debug("Created index %r", name)

        if timeout is not None and timeout != -1:
            model = self._poll_until_ready(name, timeout)

        return model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_value(value: Any) -> Any:
        """Extract .value from enum-like objects, pass through otherwise."""
        return value.value if hasattr(value, "value") else value

    def _validate_create_inputs(
        self,
        *,
        name: str,
        spec: ServerlessSpec | PodSpec | dict[str, Any],
        dimension: int | None,
        metric: Metric | str,
        vector_type: VectorType | str,
        deletion_protection: DeletionProtection | str,
    ) -> None:
        """Client-side validation for create() arguments."""
        require_non_empty("name", name)

        if spec is None:
            raise ValidationError("spec is required")

        resolved_metric = self._resolve_value(metric)
        if resolved_metric not in _VALID_METRICS:
            raise ValidationError(
                f"metric must be one of {sorted(_VALID_METRICS)}, got {resolved_metric!r}"
            )

        resolved_dp = self._resolve_value(deletion_protection)
        if resolved_dp not in _VALID_DELETION_PROTECTION:
            raise ValidationError(
                f"deletion_protection must be one of {sorted(_VALID_DELETION_PROTECTION)}, "
                f"got {resolved_dp!r}"
            )

        resolved_vt = self._resolve_value(vector_type)
        if resolved_vt != "sparse" and dimension is None:
            raise ValidationError("dimension is required for dense indexes")

    @staticmethod
    def _build_create_body(
        *,
        name: str,
        spec: ServerlessSpec | PodSpec | dict[str, Any],
        dimension: int | None,
        metric: Metric | str,
        vector_type: VectorType | str,
        deletion_protection: DeletionProtection | str,
        tags: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Build the JSON body for POST /indexes."""

        def _resolve(val: Any) -> Any:
            return val.value if hasattr(val, "value") else val

        body: dict[str, Any] = {
            "name": name,
            "metric": _resolve(metric),
            "vector_type": _resolve(vector_type),
            "deletion_protection": _resolve(deletion_protection),
        }
        if dimension is not None:
            body["dimension"] = dimension
        if tags is not None:
            body["tags"] = tags

        if isinstance(spec, ServerlessSpec):
            body["spec"] = {"serverless": {"cloud": spec.cloud, "region": spec.region}}
        elif isinstance(spec, PodSpec):
            body["spec"] = {"pod": msgspec.to_builtins(spec)}
        elif isinstance(spec, dict):
            body["spec"] = spec

        return body

    @staticmethod
    def _validate_integrated_inputs(
        *,
        name: str,
        spec: IntegratedSpec,
    ) -> None:
        """Client-side validation for integrated index creation."""
        require_non_empty("name", name)
        if not spec.cloud or not spec.cloud.strip():
            raise ValidationError("cloud is required for integrated indexes")
        if not spec.embed.model or not spec.embed.model.strip():
            raise ValidationError("embed model is required for integrated indexes")
        if not spec.embed.field_map:
            raise ValidationError("embed field_map is required for integrated indexes")

    def _build_integrated_body(
        self,
        *,
        name: str,
        spec: IntegratedSpec,
        deletion_protection: DeletionProtection | str,
        tags: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Build the JSON body for POST /indexes (integrated/model-backed)."""
        embed_body: dict[str, Any] = {
            "model": self._resolve_value(spec.embed.model),
            "field_map": spec.embed.field_map,
        }
        if spec.embed.metric is not None:
            embed_body["metric"] = self._resolve_value(spec.embed.metric)
        if spec.embed.read_parameters is not None:
            embed_body["read_parameters"] = spec.embed.read_parameters
        if spec.embed.write_parameters is not None:
            embed_body["write_parameters"] = spec.embed.write_parameters

        body: dict[str, Any] = {
            "name": name,
            "cloud": self._resolve_value(spec.cloud),
            "region": spec.region,
            "embed": embed_body,
        }

        resolved_dp = self._resolve_value(deletion_protection)
        if resolved_dp != "disabled":
            body["deletion_protection"] = resolved_dp
        if tags is not None:
            body["tags"] = tags

        return body

    def _poll_until_ready(self, name: str, timeout: int) -> IndexModel:
        """Poll describe() until the index is ready or timeout is reached."""
        start = time.monotonic()
        while True:
            idx = self.describe(name)
            if idx.status.ready:
                return idx
            if idx.status.state == "InitializationFailed":
                raise IndexInitFailedError(name)
            elapsed = time.monotonic() - start
            if elapsed + _POLL_INTERVAL_SECONDS > timeout:
                raise PineconeTimeoutError(f"Index '{name}' not ready after {timeout}s")
            time.sleep(_POLL_INTERVAL_SECONDS)
