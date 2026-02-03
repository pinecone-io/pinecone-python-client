"""Adapter functions for converting OpenAPI responses to SDK dataclasses.

This module provides centralized functions for transforming OpenAPI response
objects into SDK-specific dataclasses. These adapters isolate the SDK from
changes in the OpenAPI model structure.

The adapter functions replace duplicated parsing logic that was previously
scattered across multiple modules.
"""

from __future__ import annotations

from multiprocessing.pool import ApplyResult
from typing import TYPE_CHECKING, Any

from pinecone.adapters.protocols import (
    FetchResponseAdapter,
    QueryResponseAdapter,
    UpsertResponseAdapter,
)
from pinecone.adapters.utils import extract_response_metadata

if TYPE_CHECKING:
    from pinecone.db_data.dataclasses.fetch_response import FetchResponse
    from pinecone.db_data.dataclasses.query_response import QueryResponse
    from pinecone.db_data.dataclasses.upsert_response import UpsertResponse


def adapt_query_response(openapi_response: QueryResponseAdapter) -> QueryResponse:
    """Adapt an OpenAPI QueryResponse to the SDK QueryResponse dataclass.

    This function extracts fields from the OpenAPI response object and
    constructs an SDK-native QueryResponse dataclass. It handles:
    - Extracting matches and namespace
    - Optional usage information
    - Response metadata (headers)
    - Cleaning up deprecated 'results' field

    Args:
        openapi_response: An OpenAPI QueryResponse object from the generated code.

    Returns:
        A QueryResponse dataclass instance.

    Example:
        >>> from pinecone.adapters import adapt_query_response
        >>> sdk_response = adapt_query_response(openapi_response)
        >>> print(sdk_response.matches)
    """
    # Import at runtime to avoid circular imports
    from pinecone.db_data.dataclasses.query_response import QueryResponse as QR

    response_info = extract_response_metadata(openapi_response)

    # Remove deprecated 'results' field if present
    if hasattr(openapi_response, "_data_store"):
        openapi_response._data_store.pop("results", None)

    return QR(
        matches=openapi_response.matches,
        namespace=openapi_response.namespace or "",
        usage=openapi_response.usage
        if hasattr(openapi_response, "usage") and openapi_response.usage
        else None,
        _response_info=response_info,
    )


def adapt_upsert_response(openapi_response: UpsertResponseAdapter) -> UpsertResponse:
    """Adapt an OpenAPI UpsertResponse to the SDK UpsertResponse dataclass.

    Args:
        openapi_response: An OpenAPI UpsertResponse object from the generated code.

    Returns:
        An UpsertResponse dataclass instance.

    Example:
        >>> from pinecone.adapters import adapt_upsert_response
        >>> sdk_response = adapt_upsert_response(openapi_response)
        >>> print(sdk_response.upserted_count)
    """
    # Import at runtime to avoid circular imports
    from pinecone.db_data.dataclasses.upsert_response import UpsertResponse as UR

    response_info = extract_response_metadata(openapi_response)

    return UR(upserted_count=openapi_response.upserted_count, _response_info=response_info)


def adapt_fetch_response(openapi_response: FetchResponseAdapter) -> FetchResponse:
    """Adapt an OpenAPI FetchResponse to the SDK FetchResponse dataclass.

    This function extracts fields from the OpenAPI response object and
    constructs an SDK-native FetchResponse dataclass. It handles:
    - Converting vectors dict to SDK Vector objects
    - Optional usage information
    - Response metadata (headers)

    Args:
        openapi_response: An OpenAPI FetchResponse object from the generated code.

    Returns:
        A FetchResponse dataclass instance.

    Example:
        >>> from pinecone.adapters import adapt_fetch_response
        >>> sdk_response = adapt_fetch_response(openapi_response)
        >>> print(sdk_response.vectors)
    """
    # Import at runtime to avoid circular imports
    from pinecone.db_data.dataclasses.fetch_response import FetchResponse as FR
    from pinecone.db_data.dataclasses.vector import Vector

    response_info = extract_response_metadata(openapi_response)

    return FR(
        namespace=openapi_response.namespace or "",
        vectors={k: Vector.from_dict(v) for k, v in openapi_response.vectors.items()},
        usage=openapi_response.usage,
        _response_info=response_info,
    )


class UpsertResponseTransformer:
    """Transformer for converting ApplyResult[OpenAPIUpsertResponse] to UpsertResponse.

    This wrapper transforms the OpenAPI response to our dataclass when .get() is called,
    while delegating other methods to the underlying ApplyResult.

    Example:
        >>> transformer = UpsertResponseTransformer(async_result)
        >>> response = transformer.get()  # Returns UpsertResponse
    """

    _apply_result: ApplyResult
    """ :meta private: """

    def __init__(self, apply_result: ApplyResult) -> None:
        self._apply_result = apply_result

    def get(self, timeout: float | None = None) -> UpsertResponse:
        """Get the transformed UpsertResponse.

        Args:
            timeout: Optional timeout in seconds for the underlying result.

        Returns:
            The SDK UpsertResponse dataclass.
        """
        openapi_response = self._apply_result.get(timeout)
        return adapt_upsert_response(openapi_response)

    def __getattr__(self, name: str) -> Any:
        # Delegate other methods to the underlying ApplyResult
        return getattr(self._apply_result, name)
