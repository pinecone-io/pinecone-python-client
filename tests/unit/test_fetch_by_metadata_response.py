"""Tests for FetchByMetadataResponse model and adapter."""

from __future__ import annotations

import msgspec

from pinecone._internal.adapters.vectors_adapter import VectorsAdapter
from pinecone.models.vectors.responses import FetchByMetadataResponse


def _encode(**kwargs: object) -> bytes:
    """Encode a dict to JSON bytes using msgspec (produces camelCase-free raw JSON)."""
    return msgspec.json.encode(kwargs)


class TestFetchByMetadataResponseDecode:
    def test_decode_full_response(self) -> None:
        data = msgspec.json.encode(
            {
                "vectors": {
                    "vec1": {
                        "id": "vec1",
                        "values": [0.1, 0.2, 0.3],
                    },
                    "vec2": {
                        "id": "vec2",
                        "values": [0.4, 0.5, 0.6],
                        "metadata": {"genre": "comedy"},
                    },
                },
                "namespace": "my-ns",
                "usage": {"readUnits": 5},
                "pagination": {"next": "token-abc"},
            }
        )
        resp = msgspec.json.decode(data, type=FetchByMetadataResponse)

        assert len(resp.vectors) == 2
        assert resp.vectors["vec1"].id == "vec1"
        assert resp.vectors["vec1"].values == [0.1, 0.2, 0.3]
        assert resp.vectors["vec2"].metadata == {"genre": "comedy"}
        assert resp.namespace == "my-ns"
        assert resp.usage is not None
        assert resp.usage.read_units == 5
        assert resp.pagination is not None
        assert resp.pagination.next == "token-abc"

    def test_decode_empty_vectors(self) -> None:
        data = msgspec.json.encode({"vectors": {}})
        resp = msgspec.json.decode(data, type=FetchByMetadataResponse)

        assert resp.vectors == {}

    def test_decode_no_pagination(self) -> None:
        data = msgspec.json.encode(
            {
                "vectors": {
                    "v1": {"id": "v1", "values": [1.0]},
                },
                "namespace": "ns",
                "usage": {"readUnits": 1},
            }
        )
        resp = msgspec.json.decode(data, type=FetchByMetadataResponse)

        assert resp.pagination is None

    def test_decode_with_pagination_token(self) -> None:
        data = msgspec.json.encode(
            {
                "vectors": {},
                "pagination": {"next": "token123"},
            }
        )
        resp = msgspec.json.decode(data, type=FetchByMetadataResponse)

        assert resp.pagination is not None
        assert resp.pagination.next == "token123"

    def test_bracket_access(self) -> None:
        data = msgspec.json.encode(
            {
                "vectors": {"v1": {"id": "v1", "values": [1.0]}},
                "namespace": "test-ns",
            }
        )
        resp = msgspec.json.decode(data, type=FetchByMetadataResponse)

        assert resp["vectors"] == resp.vectors
        assert resp["namespace"] == "test-ns"

    def test_bracket_access_missing_key(self) -> None:
        data = msgspec.json.encode({"vectors": {}})
        resp = msgspec.json.decode(data, type=FetchByMetadataResponse)

        try:
            resp["nonexistent"]
            assert False, "Expected KeyError"
        except KeyError:
            pass

    def test_default_namespace_empty(self) -> None:
        data = msgspec.json.encode({"vectors": {}})
        resp = msgspec.json.decode(data, type=FetchByMetadataResponse)

        assert resp.namespace == ""


class TestVectorsAdapterFetchByMetadata:
    def test_adapter_decodes_response(self) -> None:
        data = msgspec.json.encode(
            {
                "vectors": {
                    "id1": {"id": "id1", "values": [0.1, 0.2]},
                },
                "namespace": "ns1",
                "usage": {"readUnits": 3},
                "pagination": {"next": "page2"},
            }
        )
        resp = VectorsAdapter.to_fetch_by_metadata_response(data)

        assert isinstance(resp, FetchByMetadataResponse)
        assert resp.vectors["id1"].id == "id1"
        assert resp.namespace == "ns1"
        assert resp.usage is not None
        assert resp.usage.read_units == 3
        assert resp.pagination is not None
        assert resp.pagination.next == "page2"
