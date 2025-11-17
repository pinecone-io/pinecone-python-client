"""Performance benchmarks for gRPC response parsing functions.

These tests measure the performance of parse_fetch_response and parse_query_response
to establish baselines and verify optimizations.
"""

import random
import pytest
from google.protobuf import struct_pb2

from pinecone.core.grpc.protos.db_data_2025_10_pb2 import (
    FetchResponse,
    QueryResponse,
    Vector,
    ScoredVector,
    SparseValues,
    Usage,
)
from pinecone.grpc.utils import parse_fetch_response, parse_query_response


def create_vector(id: str, dimension: int, include_sparse: bool = False) -> Vector:
    """Create a Vector protobuf message with random values."""
    values = [random.random() for _ in range(dimension)]

    # Create sparse values if needed
    sparse_values_obj = None
    if include_sparse:
        # Create sparse values with ~10% of dimension as non-zero
        sparse_size = max(1, dimension // 10)
        indices = sorted(random.sample(range(dimension), sparse_size))
        sparse_values_list = [random.random() for _ in range(sparse_size)]
        sparse_values_obj = SparseValues(indices=indices, values=sparse_values_list)

    # Add some metadata
    metadata = struct_pb2.Struct()
    metadata.update({"category": f"cat_{random.randint(1, 10)}", "score": random.random()})

    # Create vector with all fields
    if sparse_values_obj:
        vector = Vector(id=id, values=values, sparse_values=sparse_values_obj, metadata=metadata)
    else:
        vector = Vector(id=id, values=values, metadata=metadata)

    return vector


def create_scored_vector(id: str, dimension: int, include_sparse: bool = False) -> ScoredVector:
    """Create a ScoredVector protobuf message with random values."""
    values = [random.random() for _ in range(dimension)]

    # Create sparse values if needed
    sparse_values_obj = None
    if include_sparse:
        # Create sparse values with ~10% of dimension as non-zero
        sparse_size = max(1, dimension // 10)
        indices = sorted(random.sample(range(dimension), sparse_size))
        sparse_values_list = [random.random() for _ in range(sparse_size)]
        sparse_values_obj = SparseValues(indices=indices, values=sparse_values_list)

    # Add some metadata
    metadata = struct_pb2.Struct()
    metadata.update({"category": f"cat_{random.randint(1, 10)}", "score": random.random()})

    # Create scored vector with all fields
    if sparse_values_obj:
        scored_vector = ScoredVector(
            id=id,
            score=random.random(),
            values=values,
            sparse_values=sparse_values_obj,
            metadata=metadata,
        )
    else:
        scored_vector = ScoredVector(id=id, score=random.random(), values=values, metadata=metadata)

    return scored_vector


def create_fetch_response(
    num_vectors: int, dimension: int, include_sparse: bool = False
) -> FetchResponse:
    """Create a FetchResponse protobuf message with specified number of vectors."""
    vectors = {}
    for i in range(num_vectors):
        vector = create_vector(f"vec_{i}", dimension, include_sparse)
        vectors[f"vec_{i}"] = vector

    return FetchResponse(
        vectors=vectors, namespace="test_namespace", usage=Usage(read_units=num_vectors)
    )


def create_query_response(
    num_matches: int, dimension: int, include_sparse: bool = False
) -> QueryResponse:
    """Create a QueryResponse protobuf message with specified number of matches."""
    matches = [
        create_scored_vector(f"match_{i}", dimension, include_sparse) for i in range(num_matches)
    ]

    return QueryResponse(
        matches=matches, namespace="test_namespace", usage=Usage(read_units=num_matches)
    )


class TestFetchResponseParsingPerf:
    """Performance benchmarks for parse_fetch_response."""

    @pytest.mark.parametrize(
        "num_vectors,dimension",
        [
            (10, 128),
            (10, 512),
            (10, 1024),
            (100, 128),
            (100, 512),
            (100, 1024),
            (1000, 128),
            (1000, 512),
            (1000, 1024),
        ],
    )
    def test_parse_fetch_response_dense(self, benchmark, num_vectors, dimension):
        """Benchmark parse_fetch_response with dense vectors."""
        response = create_fetch_response(num_vectors, dimension, include_sparse=False)
        benchmark(parse_fetch_response, response, None)

    @pytest.mark.parametrize("num_vectors,dimension", [(10, 128), (100, 128), (1000, 128)])
    def test_parse_fetch_response_sparse(self, benchmark, num_vectors, dimension):
        """Benchmark parse_fetch_response with sparse vectors."""
        response = create_fetch_response(num_vectors, dimension, include_sparse=True)
        benchmark(parse_fetch_response, response, None)


class TestQueryResponseParsingPerf:
    """Performance benchmarks for parse_query_response."""

    @pytest.mark.parametrize(
        "num_matches,dimension",
        [
            (10, 128),
            (10, 512),
            (10, 1024),
            (100, 128),
            (100, 512),
            (100, 1024),
            (1000, 128),
            (1000, 512),
            (1000, 1024),
        ],
    )
    def test_parse_query_response_dense(self, benchmark, num_matches, dimension):
        """Benchmark parse_query_response with dense vectors."""
        response = create_query_response(num_matches, dimension, include_sparse=False)
        benchmark(parse_query_response, response, False, None)

    @pytest.mark.parametrize("num_matches,dimension", [(10, 128), (100, 128), (1000, 128)])
    def test_parse_query_response_sparse(self, benchmark, num_matches, dimension):
        """Benchmark parse_query_response with sparse vectors."""
        response = create_query_response(num_matches, dimension, include_sparse=True)
        benchmark(parse_query_response, response, False, None)
