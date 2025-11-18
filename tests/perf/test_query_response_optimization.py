"""Performance tests for parse_query_response optimizations.

This test measures the performance impact of optimizations to parse_query_response,
including metadata conversion, list pre-allocation, and other micro-optimizations.
"""

import random
import pytest
from google.protobuf import struct_pb2

from pinecone.core.grpc.protos.db_data_2025_10_pb2 import (
    QueryResponse,
    ScoredVector,
    SparseValues,
    Usage,
)
from pinecone.grpc.utils import parse_query_response


def create_scored_vector_with_metadata(
    id: str, dimension: int, include_sparse: bool = False, metadata_size: int = 2
) -> ScoredVector:
    """Create a ScoredVector protobuf message with metadata."""
    values = [random.random() for _ in range(dimension)]

    # Create sparse values if needed
    sparse_values_obj = None
    if include_sparse:
        sparse_size = max(1, dimension // 10)
        indices = sorted(random.sample(range(dimension), sparse_size))
        sparse_values_list = [random.random() for _ in range(sparse_size)]
        sparse_values_obj = SparseValues(indices=indices, values=sparse_values_list)

    # Create metadata with specified number of fields
    metadata = struct_pb2.Struct()
    metadata_dict = {}
    for i in range(metadata_size):
        metadata_dict[f"key_{i}"] = f"value_{random.randint(1, 100)}"
        if i % 3 == 0:
            metadata_dict[f"num_{i}"] = random.random()
        elif i % 3 == 1:
            metadata_dict[f"bool_{i}"] = random.choice([True, False])
    metadata.update(metadata_dict)

    # Create scored vector
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


def create_query_response_with_metadata(
    num_matches: int, dimension: int, include_sparse: bool = False, metadata_size: int = 2
) -> QueryResponse:
    """Create a QueryResponse protobuf message with matches that have metadata."""
    matches = []
    for i in range(num_matches):
        match = create_scored_vector_with_metadata(
            f"match_{i}", dimension, include_sparse, metadata_size
        )
        matches.append(match)

    return QueryResponse(
        matches=matches, namespace="test_namespace", usage=Usage(read_units=num_matches)
    )


class TestQueryResponseOptimization:
    """Performance benchmarks for parse_query_response optimizations."""

    @pytest.mark.parametrize(
        "num_matches,dimension,metadata_size",
        [
            (10, 128, 0),  # No metadata
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
    def test_parse_query_response_with_metadata(
        self, benchmark, num_matches, dimension, metadata_size
    ):
        """Benchmark parse_query_response with matches containing varying metadata."""
        response = create_query_response_with_metadata(
            num_matches, dimension, include_sparse=False, metadata_size=metadata_size
        )
        benchmark(parse_query_response, response, False, None)

    @pytest.mark.parametrize("num_matches,dimension", [(10, 128), (100, 128), (1000, 128)])
    def test_parse_query_response_sparse_with_metadata(self, benchmark, num_matches, dimension):
        """Benchmark parse_query_response with sparse vectors and metadata."""
        response = create_query_response_with_metadata(
            num_matches, dimension, include_sparse=True, metadata_size=5
        )
        benchmark(parse_query_response, response, False, None)

    @pytest.mark.parametrize(
        "num_matches,dimension",
        [(10, 512), (100, 512), (1000, 512), (10, 1024), (100, 1024), (1000, 1024)],
    )
    def test_parse_query_response_large_vectors(self, benchmark, num_matches, dimension):
        """Benchmark parse_query_response with large dimension vectors."""
        response = create_query_response_with_metadata(
            num_matches, dimension, include_sparse=False, metadata_size=2
        )
        benchmark(parse_query_response, response, False, None)

    def test_parse_query_response_empty_values(self, benchmark):
        """Benchmark parse_query_response with matches that have no values (ID-only queries)."""
        matches = []
        for i in range(100):
            metadata = struct_pb2.Struct()
            metadata.update({"category": f"cat_{i}"})
            match = ScoredVector(id=f"match_{i}", score=random.random(), metadata=metadata)
            matches.append(match)

        response = QueryResponse(matches=matches, namespace="test_namespace")
        benchmark(parse_query_response, response, False, None)
