"""Factory functions for db_control OpenAPI models.

These factories provide sensible defaults for commonly-used OpenAPI models,
allowing tests to focus on the behavior being tested rather than model construction.
"""

from typing import Optional, Dict, Any, List

from pinecone.core.openapi.db_control.models import (
    IndexModel as OpenApiIndexModel,
    IndexList as OpenApiIndexList,
    IndexModelStatus,
    CollectionList as OpenApiCollectionList,
    CollectionModel,
)


def make_index_status(
    ready: bool = True, state: str = "Ready", **overrides: Any
) -> IndexModelStatus:
    """Create an IndexModelStatus instance.

    Args:
        ready: Whether the index is ready
        state: The index state (Initializing, Ready, Terminating, etc.)
        **overrides: Additional fields to override

    Returns:
        An IndexModelStatus instance
    """
    return IndexModelStatus(ready=ready, state=state, **overrides)


def make_index_model(
    name: str = "test-index",
    host: str = "https://test-index.pinecone.io",
    status: Optional[IndexModelStatus] = None,
    deletion_protection: str = "disabled",
    dimension: int = 1536,
    metric: str = "cosine",
    spec: Optional[Dict[str, Any]] = None,
    _check_type: bool = False,
    **overrides: Any,
) -> OpenApiIndexModel:
    """Create an OpenApiIndexModel instance with sensible defaults.

    Args:
        name: Index name
        host: Index host URL
        status: Index status
        deletion_protection: "enabled" or "disabled"
        dimension: Vector dimension
        metric: Distance metric (cosine, euclidean, dotproduct)
        spec: Spec dict for serverless/pod configuration
        _check_type: Whether to enable type checking
        **overrides: Additional fields to override

    Returns:
        An OpenApiIndexModel instance
    """
    if status is None:
        status = make_index_status()

    if spec is None:
        spec = {"serverless": {"cloud": "aws", "region": "us-east-1"}}

    kwargs: Dict[str, Any] = {
        "name": name,
        "host": host,
        "status": status,
        "deletion_protection": deletion_protection,
        "dimension": dimension,
        "metric": metric,
        "spec": spec,
        "_check_type": _check_type,
    }

    kwargs.update(overrides)
    return OpenApiIndexModel(**kwargs)


def make_index_list(
    indexes: Optional[List[OpenApiIndexModel]] = None, _check_type: bool = False, **overrides: Any
) -> OpenApiIndexList:
    """Create an OpenApiIndexList instance.

    Args:
        indexes: List of index models. Defaults to empty list.
        _check_type: Whether to enable type checking
        **overrides: Additional fields to override

    Returns:
        An OpenApiIndexList instance
    """
    if indexes is None:
        indexes = []

    return OpenApiIndexList(indexes=indexes, _check_type=_check_type, **overrides)


def make_collection_model(
    name: str = "test-collection",
    status: str = "Ready",
    environment: str = "us-west1-gcp",
    size: Optional[int] = 10000,
    dimension: Optional[int] = 1536,
    record_count: Optional[int] = 1000,
    **overrides: Any,
) -> CollectionModel:
    """Create a CollectionModel instance with sensible defaults.

    Args:
        name: Collection name
        status: Collection status (Initializing, Ready, Terminating)
        environment: Environment where collection is hosted
        size: Collection size in bytes
        dimension: Vector dimension
        record_count: Number of records in the collection
        **overrides: Additional fields to override

    Returns:
        A CollectionModel instance
    """
    kwargs: Dict[str, Any] = {"name": name, "status": status, "environment": environment}

    if size is not None:
        kwargs["size"] = size
    if dimension is not None:
        kwargs["dimension"] = dimension
    if record_count is not None:
        kwargs["record_count"] = record_count

    kwargs.update(overrides)
    return CollectionModel(**kwargs)


def make_collection_list(
    collections: Optional[List[CollectionModel]] = None, **overrides: Any
) -> OpenApiCollectionList:
    """Create an OpenApiCollectionList instance.

    Args:
        collections: List of collection models. Defaults to empty list.
        **overrides: Additional fields to override

    Returns:
        An OpenApiCollectionList instance
    """
    if collections is None:
        collections = []

    return OpenApiCollectionList(collections=collections, **overrides)
