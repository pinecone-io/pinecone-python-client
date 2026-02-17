"""Test fixture factories for OpenAPI models.

This module provides factory functions that abstract the construction of OpenAPI
models used in tests. When the OpenAPI spec changes, only these factories need
to be updated rather than every test file.

Usage:
    from tests.fixtures import make_index_model, make_vector

    # Use defaults
    index = make_index_model()

    # Override specific fields
    index = make_index_model(name="custom-name", dimension=512)
"""

from tests.fixtures.db_control_models import (
    make_index_model,
    make_index_status,
    make_index_list,
    make_collection_model,
    make_collection_list,
    make_read_capacity_status,
    make_read_capacity_on_demand,
    make_read_capacity_dedicated,
)

from tests.fixtures.db_data_models import (
    make_vector,
    make_sparse_values,
    make_list_response,
    make_list_item,
    make_pagination,
    make_search_records_vector,
    make_vector_values,
    make_search_records_request_query,
    make_search_records_request_rerank,
    make_search_records_request,
    make_usage,
    make_scored_vector,
    make_openapi_query_response,
    make_openapi_upsert_response,
    make_openapi_fetch_response,
)

__all__ = [
    # db_control models
    "make_index_model",
    "make_index_status",
    "make_index_list",
    "make_collection_model",
    "make_collection_list",
    "make_read_capacity_status",
    "make_read_capacity_on_demand",
    "make_read_capacity_dedicated",
    # db_data models
    "make_vector",
    "make_sparse_values",
    "make_list_response",
    "make_list_item",
    "make_pagination",
    "make_search_records_vector",
    "make_vector_values",
    "make_search_records_request_query",
    "make_search_records_request_rerank",
    "make_search_records_request",
    # OpenAPI response models
    "make_usage",
    "make_scored_vector",
    "make_openapi_query_response",
    "make_openapi_upsert_response",
    "make_openapi_fetch_response",
]
