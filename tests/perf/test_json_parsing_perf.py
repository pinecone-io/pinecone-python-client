"""Performance benchmarks for JSON parsing of query responses.

These tests measure the performance of json.loads() vs orjson.loads() for realistic
query response payloads to evaluate potential performance improvements.
"""

import json
import random
from typing import Any

import orjson
import pytest


def create_query_response_json(
    num_matches: int,
    dimension: int,
    include_values: bool = False,
    include_metadata: bool = False,
    include_sparse: bool = False,
) -> str:
    """Create a realistic query response JSON string.

    Args:
        num_matches: Number of matches in the response.
        dimension: Vector dimension.
        include_values: Whether to include vector values.
        include_metadata: Whether to include metadata.
        include_sparse: Whether to include sparse values.

    Returns:
        JSON string representing a query response.
    """
    matches = []
    for i in range(num_matches):
        match: dict[str, Any] = {"id": f"vector-{i}", "score": random.random()}

        if include_values:
            match["values"] = [random.random() for _ in range(dimension)]

        if include_sparse:
            # Create sparse values with ~10% of dimension as non-zero
            sparse_size = max(1, dimension // 10)
            indices = sorted(random.sample(range(dimension), sparse_size))
            sparse_values = [random.random() for _ in range(sparse_size)]
            match["sparseValues"] = {"indices": indices, "values": sparse_values}

        if include_metadata:
            match["metadata"] = {
                "category": f"cat_{random.randint(1, 10)}",
                "score": random.random(),
                "name": f"item_{i}",
            }

        matches.append(match)

    response = {
        "matches": matches,
        "namespace": "test_namespace",
        "usage": {"readUnits": num_matches},
    }

    return json.dumps(response)


class TestJsonParsingPerf:
    """Performance benchmarks for JSON parsing of query responses."""

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
    def test_json_loads_minimal(self, benchmark, num_matches, dimension):
        """Benchmark json.loads() with minimal payload (no values, no metadata, no sparse)."""
        json_str = create_query_response_json(
            num_matches=num_matches,
            dimension=dimension,
            include_values=False,
            include_metadata=False,
            include_sparse=False,
        )

        def parse():
            return json.loads(json_str)

        result = benchmark(parse)
        # Verify the result is correct
        assert isinstance(result, dict)
        assert len(result["matches"]) == num_matches
        assert result["namespace"] == "test_namespace"

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
    def test_orjson_loads_minimal(self, benchmark, num_matches, dimension):
        """Benchmark orjson.loads() with minimal payload (no values, no metadata, no sparse)."""
        json_str = create_query_response_json(
            num_matches=num_matches,
            dimension=dimension,
            include_values=False,
            include_metadata=False,
            include_sparse=False,
        )

        def parse():
            return orjson.loads(json_str.encode("utf-8"))

        result = benchmark(parse)
        # Verify the result is correct
        assert isinstance(result, dict)
        assert len(result["matches"]) == num_matches
        assert result["namespace"] == "test_namespace"

    @pytest.mark.parametrize(
        "num_matches,dimension",
        [(10, 128), (10, 512), (10, 1024), (100, 128), (100, 512), (100, 1024)],
    )
    def test_json_loads_with_values(self, benchmark, num_matches, dimension):
        """Benchmark json.loads() with vector values included."""
        json_str = create_query_response_json(
            num_matches=num_matches,
            dimension=dimension,
            include_values=True,
            include_metadata=False,
            include_sparse=False,
        )

        def parse():
            return json.loads(json_str)

        result = benchmark(parse)
        # Verify the result is correct
        assert isinstance(result, dict)
        assert len(result["matches"]) == num_matches
        assert "values" in result["matches"][0]

    @pytest.mark.parametrize(
        "num_matches,dimension",
        [(10, 128), (10, 512), (10, 1024), (100, 128), (100, 512), (100, 1024)],
    )
    def test_orjson_loads_with_values(self, benchmark, num_matches, dimension):
        """Benchmark orjson.loads() with vector values included."""
        json_str = create_query_response_json(
            num_matches=num_matches,
            dimension=dimension,
            include_values=True,
            include_metadata=False,
            include_sparse=False,
        )

        def parse():
            return orjson.loads(json_str.encode("utf-8"))

        result = benchmark(parse)
        # Verify the result is correct
        assert isinstance(result, dict)
        assert len(result["matches"]) == num_matches
        assert "values" in result["matches"][0]

    @pytest.mark.parametrize("num_matches,dimension", [(10, 128), (100, 128), (1000, 128)])
    def test_json_loads_with_metadata(self, benchmark, num_matches, dimension):
        """Benchmark json.loads() with metadata included."""
        json_str = create_query_response_json(
            num_matches=num_matches,
            dimension=dimension,
            include_values=False,
            include_metadata=True,
            include_sparse=False,
        )

        def parse():
            return json.loads(json_str)

        result = benchmark(parse)
        # Verify the result is correct
        assert isinstance(result, dict)
        assert len(result["matches"]) == num_matches
        assert "metadata" in result["matches"][0]

    @pytest.mark.parametrize("num_matches,dimension", [(10, 128), (100, 128), (1000, 128)])
    def test_orjson_loads_with_metadata(self, benchmark, num_matches, dimension):
        """Benchmark orjson.loads() with metadata included."""
        json_str = create_query_response_json(
            num_matches=num_matches,
            dimension=dimension,
            include_values=False,
            include_metadata=True,
            include_sparse=False,
        )

        def parse():
            return orjson.loads(json_str.encode("utf-8"))

        result = benchmark(parse)
        # Verify the result is correct
        assert isinstance(result, dict)
        assert len(result["matches"]) == num_matches
        assert "metadata" in result["matches"][0]

    @pytest.mark.parametrize("num_matches,dimension", [(10, 128), (100, 128)])
    def test_json_loads_with_sparse(self, benchmark, num_matches, dimension):
        """Benchmark json.loads() with sparse values included."""
        json_str = create_query_response_json(
            num_matches=num_matches,
            dimension=dimension,
            include_values=False,
            include_metadata=False,
            include_sparse=True,
        )

        def parse():
            return json.loads(json_str)

        result = benchmark(parse)
        # Verify the result is correct
        assert isinstance(result, dict)
        assert len(result["matches"]) == num_matches
        assert "sparseValues" in result["matches"][0]

    @pytest.mark.parametrize("num_matches,dimension", [(10, 128), (100, 128)])
    def test_orjson_loads_with_sparse(self, benchmark, num_matches, dimension):
        """Benchmark orjson.loads() with sparse values included."""
        json_str = create_query_response_json(
            num_matches=num_matches,
            dimension=dimension,
            include_values=False,
            include_metadata=False,
            include_sparse=True,
        )

        def parse():
            return orjson.loads(json_str.encode("utf-8"))

        result = benchmark(parse)
        # Verify the result is correct
        assert isinstance(result, dict)
        assert len(result["matches"]) == num_matches
        assert "sparseValues" in result["matches"][0]

    @pytest.mark.parametrize("num_matches,dimension", [(10, 128), (100, 128)])
    def test_json_loads_full(self, benchmark, num_matches, dimension):
        """Benchmark json.loads() with all fields (values, metadata, sparse)."""
        json_str = create_query_response_json(
            num_matches=num_matches,
            dimension=dimension,
            include_values=True,
            include_metadata=True,
            include_sparse=True,
        )

        def parse():
            return json.loads(json_str)

        result = benchmark(parse)
        # Verify the result is correct
        assert isinstance(result, dict)
        assert len(result["matches"]) == num_matches
        match = result["matches"][0]
        assert "values" in match
        assert "metadata" in match
        assert "sparseValues" in match

    @pytest.mark.parametrize("num_matches,dimension", [(10, 128), (100, 128)])
    def test_orjson_loads_full(self, benchmark, num_matches, dimension):
        """Benchmark orjson.loads() with all fields (values, metadata, sparse)."""
        json_str = create_query_response_json(
            num_matches=num_matches,
            dimension=dimension,
            include_values=True,
            include_metadata=True,
            include_sparse=True,
        )

        def parse():
            return orjson.loads(json_str.encode("utf-8"))

        result = benchmark(parse)
        # Verify the result is correct
        assert isinstance(result, dict)
        assert len(result["matches"]) == num_matches
        match = result["matches"][0]
        assert "values" in match
        assert "metadata" in match
        assert "sparseValues" in match
