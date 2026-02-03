"""Protocol definitions for the adapter layer.

This module defines formal Protocol interfaces that specify the contract between
generated OpenAPI models and SDK adapter code. These protocols make it explicit
what properties and methods the SDK code depends on from the OpenAPI models,
enabling:

- Type safety with static type checking (mypy)
- Clear documentation of adapter dependencies
- Flexibility to change OpenAPI model implementations
- Better testability through protocol-based mocking

Each protocol corresponds to an OpenAPI model type that adapters consume. The
protocols define only the minimal interface required by adapter functions,
isolating SDK code from the full complexity of generated models.

Usage:
    >>> from pinecone.adapters.protocols import QueryResponseAdapter
    >>> def adapt_query(response: QueryResponseAdapter) -> QueryResponse:
    ...     return QueryResponse(matches=response.matches)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pinecone.core.openapi.db_data.models import ScoredVector, Usage
    from pinecone.core.openapi.db_control.model.index_model_status import IndexModelStatus


class QueryResponseAdapter(Protocol):
    """Protocol for OpenAPI QueryResponse objects used in adapters.

    This protocol defines the minimal interface that SDK code depends on when
    adapting an OpenAPI QueryResponse to the SDK QueryResponse dataclass.

    Attributes:
        matches: List of scored vectors returned by the query.
        namespace: The namespace that was queried.
        usage: Optional usage statistics for the query operation.
        _data_store: Internal data storage (for accessing raw response data).
        _response_info: Response metadata including headers.
    """

    matches: list[ScoredVector]
    namespace: str | None
    usage: Usage | None
    _data_store: dict[str, Any]
    _response_info: Any


class UpsertResponseAdapter(Protocol):
    """Protocol for OpenAPI UpsertResponse objects used in adapters.

    This protocol defines the minimal interface that SDK code depends on when
    adapting an OpenAPI UpsertResponse to the SDK UpsertResponse dataclass.

    Attributes:
        upserted_count: Number of vectors that were successfully upserted.
        _response_info: Response metadata including headers.
    """

    upserted_count: int
    _response_info: Any


class FetchResponseAdapter(Protocol):
    """Protocol for OpenAPI FetchResponse objects used in adapters.

    This protocol defines the minimal interface that SDK code depends on when
    adapting an OpenAPI FetchResponse to the SDK FetchResponse dataclass.

    Attributes:
        namespace: The namespace from which vectors were fetched (optional).
        vectors: Dictionary mapping vector IDs to Vector objects.
        usage: Optional usage statistics for the fetch operation.
        _response_info: Response metadata including headers.
    """

    namespace: str | None
    vectors: dict[str, Any]
    usage: Usage | None
    _response_info: Any


class IndexModelAdapter(Protocol):
    """Protocol for OpenAPI IndexModel objects used in adapters.

    This protocol defines the minimal interface that SDK code depends on when
    working with OpenAPI IndexModel objects. The IndexModel wrapper class
    provides additional functionality on top of this protocol.

    Attributes:
        name: The name of the index.
        dimension: The dimensionality of vectors in the index.
        metric: The distance metric used for similarity search.
        host: The host URL for the index.
        spec: The index specification (serverless, pod, or BYOC).
        status: The current status of the index.
        _data_store: Internal data storage (for accessing raw response data).
        _configuration: OpenAPI configuration object.
        _path_to_item: Path to this item in the response tree.
    """

    name: str
    dimension: int
    metric: str
    host: str
    spec: Any
    status: IndexModelStatus
    _data_store: dict[str, Any]
    _configuration: Any
    _path_to_item: tuple[str, ...] | list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert the index model to a dictionary representation.

        Returns:
            Dictionary representation of the index model.
        """
        ...


class IndexStatusAdapter(Protocol):
    """Protocol for IndexModelStatus objects used in adapters.

    This protocol defines the minimal interface that SDK code depends on when
    working with index status information.

    Attributes:
        ready: Whether the index is ready to serve requests.
        state: The current state of the index (e.g., 'Ready', 'Initializing').
    """

    ready: bool
    state: str
