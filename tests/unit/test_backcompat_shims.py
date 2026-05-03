"""Import round-trip tests for pre-rewrite backcompat shim modules."""

from __future__ import annotations

import importlib

import pytest

# (legacy_path, symbol_name, canonical_path)
_REEXPORT_TRIPLES = [
    (
        "pinecone.core.openapi.db_data.models",
        "DescribeIndexStatsResponse",
        "pinecone.models.vectors.responses",
    ),
    (
        "pinecone.db_control.models.byoc_spec",
        "ByocSpec",
        "pinecone.models.indexes.specs",
    ),
    (
        "pinecone.db_data.dataclasses.fetch_by_metadata_response",
        "FetchByMetadataResponse",
        "pinecone.models.vectors.responses",
    ),
    (
        "pinecone.db_data.dataclasses.fetch_by_metadata_response",
        "Pagination",
        "pinecone.models.vectors.responses",
    ),
    (
        "pinecone.db_data.dataclasses.fetch_response",
        "FetchResponse",
        "pinecone.models.vectors.responses",
    ),
    (
        "pinecone.db_data.dataclasses.query_response",
        "QueryResponse",
        "pinecone.models.vectors.responses",
    ),
    (
        "pinecone.db_data.dataclasses.sparse_values",
        "SparseValues",
        "pinecone.models.vectors.sparse",
    ),
    (
        "pinecone.db_data.dataclasses.update_response",
        "UpdateResponse",
        "pinecone.models.vectors.responses",
    ),
    (
        "pinecone.db_data.dataclasses.upsert_response",
        "UpsertResponse",
        "pinecone.models.vectors.responses",
    ),
    (
        "pinecone.db_data.dataclasses.vector",
        "Vector",
        "pinecone.models.vectors.vector",
    ),
    (
        "pinecone.db_data.query_results_aggregator",
        "QueryResultsAggregator",
        "pinecone.models.vectors.query_aggregator",
    ),
    (
        "pinecone.db_data.query_results_aggregator",
        "QueryResultsAggregatorInvalidTopKError",
        "pinecone.models.vectors.query_aggregator",
    ),
]


@pytest.mark.parametrize("legacy_path,symbol_name,canonical_path", _REEXPORT_TRIPLES)
def test_shim_reexports_canonical(legacy_path: str, symbol_name: str, canonical_path: str) -> None:
    legacy_module = importlib.import_module(legacy_path)
    canonical_module = importlib.import_module(canonical_path)
    assert getattr(legacy_module, symbol_name) is getattr(canonical_module, symbol_name)


@pytest.mark.parametrize("legacy_path,symbol_name,canonical_path", _REEXPORT_TRIPLES)
def test_shim_module_has_all_matching_reexport(
    legacy_path: str, symbol_name: str, canonical_path: str
) -> None:
    legacy_module = importlib.import_module(legacy_path)
    assert symbol_name in legacy_module.__all__
    assert all(hasattr(legacy_module, name) for name in legacy_module.__all__)


class TestLegacyListResponse:
    def test_importable(self) -> None:
        from pinecone.db_control.models.list_response import ListResponse, Pagination

        assert ListResponse.__name__ == "ListResponse"
        assert Pagination.__name__ == "Pagination"

    def test_pagination_instantiation_and_access(self) -> None:
        from pinecone.db_control.models.list_response import Pagination

        p = Pagination(next="token-abc")
        assert p.next == "token-abc"

    def test_list_response_instantiation_and_access(self) -> None:
        from pinecone.db_control.models.list_response import ListResponse

        lr = ListResponse(namespace="ns", vectors=[], pagination=None)
        assert lr.namespace == "ns"
        assert lr.vectors == []
        assert lr.pagination is None

    def test_list_response_with_pagination(self) -> None:
        from pinecone.db_control.models.list_response import ListResponse, Pagination

        p = Pagination(next="t")
        lr = ListResponse(namespace="ns", vectors=[], pagination=p)
        assert lr.pagination is not None
        assert lr.pagination.next == "t"


class TestDbDataDataclassesPackage:
    def test_package_exports_all_symbols(self) -> None:
        import pinecone.db_data.dataclasses as pkg

        for name in pkg.__all__:
            assert hasattr(pkg, name), f"__all__ lists {name!r} but it is not an attribute"
