"""Factory functions for db_data OpenAPI models.

These factories provide sensible defaults for commonly-used OpenAPI models,
allowing tests to focus on the behavior being tested rather than model construction.
"""

from typing import Optional, Dict, Any, List

from pinecone.core.openapi.db_data.models import (
    Vector as OpenApiVector,
    SparseValues as OpenApiSparseValues,
    ListResponse,
    ListItem,
    Pagination,
    SearchRecordsVector,
    VectorValues,
    SearchRecordsRequestQuery,
    SearchRecordsRequestRerank,
    SearchRecordsRequest,
)


def make_sparse_values(
    indices: Optional[List[int]] = None, values: Optional[List[float]] = None, **overrides: Any
) -> OpenApiSparseValues:
    """Create an OpenApiSparseValues instance.

    Args:
        indices: Sparse vector indices. Defaults to [0, 2].
        values: Sparse vector values. Defaults to [0.1, 0.3].
        **overrides: Additional fields to override

    Returns:
        An OpenApiSparseValues instance
    """
    if indices is None:
        indices = [0, 2]
    if values is None:
        values = [0.1, 0.3]

    return OpenApiSparseValues(indices=indices, values=values, **overrides)


def make_vector(
    id: str = "vec1",
    values: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    sparse_values: Optional[OpenApiSparseValues] = None,
    **overrides: Any,
) -> OpenApiVector:
    """Create an OpenApiVector instance with sensible defaults.

    Args:
        id: Vector ID
        values: Dense vector values. Defaults to [0.1, 0.2, 0.3].
        metadata: Vector metadata
        sparse_values: Sparse vector values
        **overrides: Additional fields to override

    Returns:
        An OpenApiVector instance
    """
    if values is None:
        values = [0.1, 0.2, 0.3]

    kwargs: Dict[str, Any] = {"id": id, "values": values}

    if metadata is not None:
        kwargs["metadata"] = metadata
    if sparse_values is not None:
        kwargs["sparse_values"] = sparse_values

    kwargs.update(overrides)
    return OpenApiVector(**kwargs)


def make_list_item(id: str = "vec1", **overrides: Any) -> ListItem:
    """Create a ListItem instance.

    Args:
        id: Vector ID
        **overrides: Additional fields to override

    Returns:
        A ListItem instance
    """
    return ListItem(id=id, **overrides)


def make_pagination(
    next: str = "next-token", _check_type: bool = False, **overrides: Any
) -> Pagination:
    """Create a Pagination instance.

    Args:
        next: Pagination token for the next page
        _check_type: Whether to enable type checking
        **overrides: Additional fields to override

    Returns:
        A Pagination instance
    """
    return Pagination(next=next, _check_type=_check_type, **overrides)


def make_list_response(
    vectors: Optional[List[ListItem]] = None,
    namespace: str = "",
    pagination: Optional[Pagination] = None,
    _check_type: bool = False,
    **overrides: Any,
) -> ListResponse:
    """Create a ListResponse instance.

    Args:
        vectors: List of vector items. Defaults to empty list.
        namespace: Namespace of the vectors
        pagination: Pagination information
        _check_type: Whether to enable type checking
        **overrides: Additional fields to override

    Returns:
        A ListResponse instance
    """
    if vectors is None:
        vectors = []

    return ListResponse(
        vectors=vectors,
        namespace=namespace,
        pagination=pagination,
        _check_type=_check_type,
        **overrides,
    )


def make_vector_values(values: Optional[List[float]] = None) -> VectorValues:
    """Create a VectorValues instance.

    Args:
        values: Vector values. Defaults to [0.1, 0.2, 0.3].

    Returns:
        A VectorValues instance
    """
    if values is None:
        values = [0.1, 0.2, 0.3]
    return VectorValues(values)


_UNSET = object()  # Sentinel for distinguishing None from unset


def make_search_records_vector(
    values: Optional[List[float]] = _UNSET,  # type: ignore[assignment]
    sparse_indices: Optional[List[int]] = None,
    sparse_values: Optional[List[float]] = None,
    **overrides: Any,
) -> SearchRecordsVector:
    """Create a SearchRecordsVector instance.

    Args:
        values: Dense vector values. Defaults to [0.1, 0.2, 0.3]. Pass None to omit.
        sparse_indices: Sparse vector indices
        sparse_values: Sparse vector values
        **overrides: Additional fields to override

    Returns:
        A SearchRecordsVector instance
    """
    kwargs: Dict[str, Any] = {}

    if values is _UNSET:
        # Default to a standard vector
        kwargs["values"] = VectorValues([0.1, 0.2, 0.3])
    elif values is not None:
        kwargs["values"] = VectorValues(values)
    # If values is None, don't include it (sparse-only case)

    if sparse_indices is not None:
        kwargs["sparse_indices"] = sparse_indices
    if sparse_values is not None:
        kwargs["sparse_values"] = sparse_values

    kwargs.update(overrides)
    return SearchRecordsVector(**kwargs)


def make_search_records_request_query(
    top_k: int = 10,
    inputs: Optional[Dict[str, Any]] = None,
    vector: Optional[SearchRecordsVector] = None,
    filter: Optional[Dict[str, Any]] = None,
    id: Optional[str] = None,
    **overrides: Any,
) -> SearchRecordsRequestQuery:
    """Create a SearchRecordsRequestQuery instance.

    Args:
        top_k: Number of results to return
        inputs: Input data for embedding
        vector: Query vector
        filter: Metadata filter
        id: Query by ID
        **overrides: Additional fields to override

    Returns:
        A SearchRecordsRequestQuery instance
    """
    kwargs: Dict[str, Any] = {"top_k": top_k}

    if inputs is not None:
        kwargs["inputs"] = inputs
    if vector is not None:
        kwargs["vector"] = vector
    if filter is not None:
        kwargs["filter"] = filter
    if id is not None:
        kwargs["id"] = id

    kwargs.update(overrides)
    return SearchRecordsRequestQuery(**kwargs)


def make_search_records_request_rerank(
    model: str = "bge-reranker-v2-m3",
    rank_fields: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    parameters: Optional[Dict[str, Any]] = None,
    query: Optional[str] = None,
    **overrides: Any,
) -> SearchRecordsRequestRerank:
    """Create a SearchRecordsRequestRerank instance.

    Args:
        model: Reranking model name
        rank_fields: Fields to use for reranking
        top_n: Number of results to return after reranking
        parameters: Additional model parameters
        query: Optional query override for reranking
        **overrides: Additional fields to override

    Returns:
        A SearchRecordsRequestRerank instance
    """
    if rank_fields is None:
        rank_fields = ["text"]

    kwargs: Dict[str, Any] = {"model": model, "rank_fields": rank_fields}

    if top_n is not None:
        kwargs["top_n"] = top_n
    if parameters is not None:
        kwargs["parameters"] = parameters
    if query is not None:
        kwargs["query"] = query

    kwargs.update(overrides)
    return SearchRecordsRequestRerank(**kwargs)


def make_search_records_request(
    query: Optional[SearchRecordsRequestQuery] = None,
    fields: Optional[List[str]] = None,
    rerank: Optional[SearchRecordsRequestRerank] = None,
    **overrides: Any,
) -> SearchRecordsRequest:
    """Create a SearchRecordsRequest instance.

    Args:
        query: The search query
        fields: Fields to return in results
        rerank: Reranking configuration
        **overrides: Additional fields to override

    Returns:
        A SearchRecordsRequest instance
    """
    if query is None:
        query = make_search_records_request_query()
    if fields is None:
        fields = ["*"]

    kwargs: Dict[str, Any] = {"query": query, "fields": fields}

    if rerank is not None:
        kwargs["rerank"] = rerank

    kwargs.update(overrides)
    return SearchRecordsRequest(**kwargs)
