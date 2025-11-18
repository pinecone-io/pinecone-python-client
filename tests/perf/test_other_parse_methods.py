"""Performance tests for other parse methods.

This test measures the performance of parse_fetch_by_metadata_response,
parse_list_namespaces_response, parse_stats_response, and other parse methods.
"""

import random
import pytest
from google.protobuf import struct_pb2

from pinecone.core.grpc.protos.db_data_2025_10_pb2 import (
    FetchByMetadataResponse,
    ListNamespacesResponse,
    DescribeIndexStatsResponse,
    UpsertResponse,
    Vector,
    SparseValues,
    Usage,
    Pagination,
    NamespaceDescription as ProtoNamespaceDescription,
    NamespaceSummary,
)
from pinecone.grpc.utils import (
    parse_fetch_by_metadata_response,
    parse_list_namespaces_response,
    parse_stats_response,
    parse_upsert_response,
    parse_update_response,
    parse_namespace_description,
)
from pinecone.core.grpc.protos.db_data_2025_10_pb2 import IndexedFields as ProtoIndexedFields


def create_vector_for_fetch_by_metadata(
    id: str, dimension: int, include_sparse: bool = False, metadata_size: int = 2
) -> Vector:
    """Create a Vector protobuf message with metadata for fetch_by_metadata."""
    values = [random.random() for _ in range(dimension)]

    sparse_values_obj = None
    if include_sparse:
        sparse_size = max(1, dimension // 10)
        indices = sorted(random.sample(range(dimension), sparse_size))
        sparse_values_list = [random.random() for _ in range(sparse_size)]
        sparse_values_obj = SparseValues(indices=indices, values=sparse_values_list)

    metadata = struct_pb2.Struct()
    metadata_dict = {}
    for i in range(metadata_size):
        metadata_dict[f"key_{i}"] = f"value_{random.randint(1, 100)}"
        if i % 3 == 0:
            metadata_dict[f"num_{i}"] = random.random()
    metadata.update(metadata_dict)

    if sparse_values_obj:
        return Vector(id=id, values=values, sparse_values=sparse_values_obj, metadata=metadata)
    else:
        return Vector(id=id, values=values, metadata=metadata)


def create_fetch_by_metadata_response_with_metadata(
    num_vectors: int, dimension: int, include_sparse: bool = False, metadata_size: int = 2
) -> FetchByMetadataResponse:
    """Create a FetchByMetadataResponse protobuf message with vectors that have metadata."""
    vectors = {}
    for i in range(num_vectors):
        vector = create_vector_for_fetch_by_metadata(
            f"vec_{i}", dimension, include_sparse, metadata_size
        )
        vectors[f"vec_{i}"] = vector

    pagination = Pagination(next="next_token") if num_vectors > 10 else None

    return FetchByMetadataResponse(
        vectors=vectors,
        namespace="test_namespace",
        usage=Usage(read_units=num_vectors),
        pagination=pagination,
    )


def create_list_namespaces_response(num_namespaces: int) -> ListNamespacesResponse:
    """Create a ListNamespacesResponse protobuf message."""
    namespaces = []
    for i in range(num_namespaces):
        indexed_fields = None
        if i % 2 == 0:  # Some namespaces have indexed fields
            indexed_fields = ProtoIndexedFields(fields=[f"field_{j}" for j in range(3)])

        namespace = ProtoNamespaceDescription(
            name=f"namespace_{i}",
            record_count=random.randint(100, 10000),
            indexed_fields=indexed_fields,
        )
        namespaces.append(namespace)

    pagination = Pagination(next="next_token") if num_namespaces > 10 else None

    return ListNamespacesResponse(
        namespaces=namespaces, pagination=pagination, total_count=num_namespaces
    )


def create_stats_response(
    num_namespaces: int, dimension: int | None = 128
) -> DescribeIndexStatsResponse:
    """Create a DescribeIndexStatsResponse protobuf message."""
    namespaces = {}
    for i in range(num_namespaces):
        namespaces[f"namespace_{i}"] = NamespaceSummary(vector_count=random.randint(100, 10000))

    return DescribeIndexStatsResponse(
        namespaces=namespaces,
        dimension=dimension,
        index_fullness=random.random(),
        total_vector_count=sum(ns.vector_count for ns in namespaces.values()),
    )


class TestFetchByMetadataResponseOptimization:
    """Performance benchmarks for parse_fetch_by_metadata_response optimizations."""

    @pytest.mark.parametrize(
        "num_vectors,dimension,metadata_size",
        [
            (10, 128, 0),
            (10, 128, 2),
            (10, 128, 10),
            (100, 128, 0),
            (100, 128, 2),
            (100, 128, 10),
            (1000, 128, 0),
            (1000, 128, 2),
            (1000, 128, 10),
        ],
    )
    def test_parse_fetch_by_metadata_response_with_metadata(
        self, benchmark, num_vectors, dimension, metadata_size
    ):
        """Benchmark parse_fetch_by_metadata_response with vectors containing varying metadata."""
        response = create_fetch_by_metadata_response_with_metadata(
            num_vectors, dimension, include_sparse=False, metadata_size=metadata_size
        )
        benchmark(parse_fetch_by_metadata_response, response, None)

    @pytest.mark.parametrize("num_vectors,dimension", [(10, 128), (100, 128), (1000, 128)])
    def test_parse_fetch_by_metadata_response_sparse(self, benchmark, num_vectors, dimension):
        """Benchmark parse_fetch_by_metadata_response with sparse vectors."""
        response = create_fetch_by_metadata_response_with_metadata(
            num_vectors, dimension, include_sparse=True, metadata_size=5
        )
        benchmark(parse_fetch_by_metadata_response, response, None)


class TestListNamespacesResponseOptimization:
    """Performance benchmarks for parse_list_namespaces_response optimizations."""

    @pytest.mark.parametrize("num_namespaces", [10, 50, 100, 500, 1000])
    def test_parse_list_namespaces_response(self, benchmark, num_namespaces):
        """Benchmark parse_list_namespaces_response with varying numbers of namespaces."""
        response = create_list_namespaces_response(num_namespaces)
        benchmark(parse_list_namespaces_response, response)


class TestStatsResponseOptimization:
    """Performance benchmarks for parse_stats_response optimizations."""

    @pytest.mark.parametrize("num_namespaces", [10, 50, 100, 500, 1000])
    def test_parse_stats_response(self, benchmark, num_namespaces):
        """Benchmark parse_stats_response with varying numbers of namespaces."""
        response = create_stats_response(num_namespaces, dimension=128)
        benchmark(parse_stats_response, response)

    def test_parse_stats_response_sparse_index(self, benchmark):
        """Benchmark parse_stats_response for sparse index (no dimension)."""
        response = create_stats_response(100, dimension=None)
        benchmark(parse_stats_response, response)


class TestSimpleParseMethods:
    """Performance benchmarks for simple parse methods."""

    def test_parse_upsert_response(self, benchmark):
        """Benchmark parse_upsert_response."""
        response = UpsertResponse(upserted_count=1000)
        benchmark(parse_upsert_response, response, False, None)

    def test_parse_update_response(self, benchmark):
        """Benchmark parse_update_response."""
        from pinecone.core.grpc.protos.db_data_2025_10_pb2 import (
            UpdateResponse as ProtoUpdateResponse,
        )

        response = ProtoUpdateResponse(matched_records=500)
        benchmark(parse_update_response, response, False, None)

    def test_parse_namespace_description(self, benchmark):
        """Benchmark parse_namespace_description."""
        indexed_fields = ProtoIndexedFields(fields=["field1", "field2", "field3"])
        response = ProtoNamespaceDescription(
            name="test_namespace", record_count=5000, indexed_fields=indexed_fields
        )
        benchmark(parse_namespace_description, response, None)
