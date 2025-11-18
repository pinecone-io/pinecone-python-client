"""Performance tests for parse_fetch_response optimizations.

This test measures the performance impact of optimizations to parse_fetch_response,
specifically the _struct_to_dict optimization vs json_format.MessageToDict.
"""

import random
import pytest
from google.protobuf import struct_pb2

from pinecone.core.grpc.protos.db_data_2025_10_pb2 import FetchResponse, Vector, Usage
from pinecone.grpc.utils import parse_fetch_response, _struct_to_dict
from google.protobuf import json_format


def create_vector_with_metadata(id: str, dimension: int, metadata_size: int = 2) -> Vector:
    """Create a Vector protobuf message with metadata."""
    values = [random.random() for _ in range(dimension)]

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

    return Vector(id=id, values=values, metadata=metadata)


def create_fetch_response_with_metadata(
    num_vectors: int, dimension: int, metadata_size: int = 2
) -> FetchResponse:
    """Create a FetchResponse protobuf message with vectors that have metadata."""
    vectors = {}
    for i in range(num_vectors):
        vector = create_vector_with_metadata(f"vec_{i}", dimension, metadata_size)
        vectors[f"vec_{i}"] = vector

    return FetchResponse(
        vectors=vectors, namespace="test_namespace", usage=Usage(read_units=num_vectors)
    )


class TestFetchResponseOptimization:
    """Performance benchmarks for parse_fetch_response optimizations."""

    @pytest.mark.parametrize(
        "num_vectors,dimension,metadata_size",
        [
            (10, 128, 2),
            (10, 128, 10),
            (100, 128, 2),
            (100, 128, 10),
            (1000, 128, 2),
            (1000, 128, 10),
        ],
    )
    def test_parse_fetch_response_with_metadata(
        self, benchmark, num_vectors, dimension, metadata_size
    ):
        """Benchmark parse_fetch_response with vectors containing metadata."""
        response = create_fetch_response_with_metadata(num_vectors, dimension, metadata_size)
        benchmark(parse_fetch_response, response, None)

    def test_struct_to_dict_vs_message_to_dict(self, benchmark):
        """Compare _struct_to_dict vs json_format.MessageToDict performance."""
        # Create a struct with various value types
        struct = struct_pb2.Struct()
        struct.update(
            {
                "string_field": "test_value",
                "number_field": 123.456,
                "bool_field": True,
                "list_field": [1, 2, 3, "four", 5.0],
                "nested": {"inner": "value", "num": 42},
            }
        )

        # Benchmark our optimized version
        result_optimized = benchmark(_struct_to_dict, struct)

        # Verify correctness by comparing with MessageToDict
        result_standard = json_format.MessageToDict(struct)
        assert result_optimized == result_standard, "Results don't match!"

    @pytest.mark.parametrize("num_fields", [1, 5, 10, 20, 50])
    def test_struct_to_dict_scaling(self, benchmark, num_fields):
        """Test how _struct_to_dict performance scales with number of fields."""
        struct = struct_pb2.Struct()
        metadata_dict = {}
        for i in range(num_fields):
            metadata_dict[f"key_{i}"] = f"value_{i}"
            if i % 2 == 0:
                metadata_dict[f"num_{i}"] = float(i)
        struct.update(metadata_dict)

        result = benchmark(_struct_to_dict, struct)
        # We add num_fields string fields, plus (num_fields + 1) // 2 number fields (for even indices: 0, 2, 4, ...)
        expected_fields = num_fields + ((num_fields + 1) // 2)
        assert len(result) == expected_fields
