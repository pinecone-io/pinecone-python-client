"""Performance tests comparing orjson vs standard json library.

These tests measure the performance improvements from using orjson
for JSON serialization and deserialization in REST API requests/responses.
"""

import json
import random

import orjson
import pytest


def create_vector_payload(num_vectors: int, dimension: int) -> list[dict]:
    """Create a typical upsert payload with vectors."""
    vectors = []
    for i in range(num_vectors):
        vector = {
            "id": f"vec_{i}",
            "values": [random.random() for _ in range(dimension)],
            "metadata": {
                "category": f"cat_{i % 10}",
                "score": random.randint(0, 100),
                "tags": [f"tag_{j}" for j in range(3)],
            },
        }
        vectors.append(vector)
    return vectors


def create_query_response(num_matches: int, dimension: int, include_values: bool = True) -> dict:
    """Create a typical query response payload."""
    matches = []
    for i in range(num_matches):
        match = {
            "id": f"vec_{i}",
            "score": random.random(),
            "metadata": {"category": f"cat_{i % 10}", "score": random.randint(0, 100)},
        }
        if include_values:
            match["values"] = [random.random() for _ in range(dimension)]
        matches.append(match)
    return {"matches": matches}


class TestOrjsonSerialization:
    """Benchmark orjson.dumps() vs json.dumps()."""

    @pytest.mark.parametrize("num_vectors,dimension", [(10, 128), (100, 128), (100, 512)])
    def test_json_dumps_vectors(self, benchmark, num_vectors, dimension):
        """Benchmark json.dumps() for vector payloads."""
        payload = create_vector_payload(num_vectors, dimension)
        result = benchmark(json.dumps, payload)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize("num_vectors,dimension", [(10, 128), (100, 128), (100, 512)])
    def test_orjson_dumps_vectors(self, benchmark, num_vectors, dimension):
        """Benchmark orjson.dumps() for vector payloads."""
        payload = create_vector_payload(num_vectors, dimension)
        result = benchmark(orjson.dumps, payload)
        assert isinstance(result, bytes)
        assert len(result) > 0

    @pytest.mark.parametrize("num_matches,dimension", [(10, 128), (100, 128), (1000, 128)])
    def test_json_dumps_query_response(self, benchmark, num_matches, dimension):
        """Benchmark json.dumps() for query responses."""
        payload = create_query_response(num_matches, dimension)
        result = benchmark(json.dumps, payload)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize("num_matches,dimension", [(10, 128), (100, 128), (1000, 128)])
    def test_orjson_dumps_query_response(self, benchmark, num_matches, dimension):
        """Benchmark orjson.dumps() for query responses."""
        payload = create_query_response(num_matches, dimension)
        result = benchmark(orjson.dumps, payload)
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestOrjsonDeserialization:
    """Benchmark orjson.loads() vs json.loads()."""

    @pytest.mark.parametrize("num_vectors,dimension", [(10, 128), (100, 128), (100, 512)])
    def test_json_loads_vectors(self, benchmark, num_vectors, dimension):
        """Benchmark json.loads() for vector payloads."""
        payload = create_vector_payload(num_vectors, dimension)
        json_str = json.dumps(payload)
        result = benchmark(json.loads, json_str)
        assert isinstance(result, list)
        assert len(result) == num_vectors

    @pytest.mark.parametrize("num_vectors,dimension", [(10, 128), (100, 128), (100, 512)])
    def test_orjson_loads_vectors(self, benchmark, num_vectors, dimension):
        """Benchmark orjson.loads() for vector payloads."""
        payload = create_vector_payload(num_vectors, dimension)
        json_bytes = json.dumps(payload).encode("utf-8")
        result = benchmark(orjson.loads, json_bytes)
        assert isinstance(result, list)
        assert len(result) == num_vectors

    @pytest.mark.parametrize("num_matches,dimension", [(10, 128), (100, 128), (1000, 128)])
    def test_json_loads_query_response(self, benchmark, num_matches, dimension):
        """Benchmark json.loads() for query responses."""
        payload = create_query_response(num_matches, dimension)
        json_str = json.dumps(payload)
        result = benchmark(json.loads, json_str)
        assert isinstance(result, dict)
        assert len(result["matches"]) == num_matches

    @pytest.mark.parametrize("num_matches,dimension", [(10, 128), (100, 128), (1000, 128)])
    def test_orjson_loads_query_response(self, benchmark, num_matches, dimension):
        """Benchmark orjson.loads() for query responses."""
        payload = create_query_response(num_matches, dimension)
        json_bytes = json.dumps(payload).encode("utf-8")
        result = benchmark(orjson.loads, json_bytes)
        assert isinstance(result, dict)
        assert len(result["matches"]) == num_matches

    @pytest.mark.parametrize("num_matches,dimension", [(10, 128), (100, 128), (1000, 128)])
    def test_orjson_loads_from_string(self, benchmark, num_matches, dimension):
        """Benchmark orjson.loads() with string input (like from decoded response)."""
        payload = create_query_response(num_matches, dimension)
        json_str = json.dumps(payload)
        result = benchmark(orjson.loads, json_str)
        assert isinstance(result, dict)
        assert len(result["matches"]) == num_matches


class TestRoundTrip:
    """Benchmark complete round-trip serialization/deserialization."""

    @pytest.mark.parametrize("num_vectors,dimension", [(10, 128), (100, 128)])
    def test_json_round_trip(self, benchmark, num_vectors, dimension):
        """Benchmark json round-trip (dumps + loads)."""

        def round_trip(payload):
            json_str = json.dumps(payload)
            return json.loads(json_str)

        payload = create_vector_payload(num_vectors, dimension)
        result = benchmark(round_trip, payload)
        assert isinstance(result, list)
        assert len(result) == num_vectors

    @pytest.mark.parametrize("num_vectors,dimension", [(10, 128), (100, 128)])
    def test_orjson_round_trip(self, benchmark, num_vectors, dimension):
        """Benchmark orjson round-trip (dumps + loads)."""

        def round_trip(payload):
            json_bytes = orjson.dumps(payload)
            return orjson.loads(json_bytes)

        payload = create_vector_payload(num_vectors, dimension)
        result = benchmark(round_trip, payload)
        assert isinstance(result, list)
        assert len(result) == num_vectors
