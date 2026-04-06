"""Unit tests for Index.upsert_records() method."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from pinecone import Index
from pinecone.errors.exceptions import ValidationError
from pinecone.models.vectors.responses import UpsertRecordsResponse

INDEX_HOST = "my-index-abc123.svc.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
UPSERT_URL = f"{INDEX_HOST_HTTPS}/records/namespaces/test-ns/upsert"


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


class TestUpsertRecords:
    """Tests for Index.upsert_records() — integrated inference upsert via NDJSON."""

    @respx.mock
    def test_upsert_records_basic(self) -> None:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(201),
        )
        idx = _make_index()
        result = idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}, {"_id": "r2", "text": "world"}],
        )

        assert isinstance(result, UpsertRecordsResponse)
        assert result.record_count == 2

    @respx.mock
    def test_upsert_records_ndjson_content_type(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(201),
        )
        idx = _make_index()
        idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}],
        )

        request = route.calls.last.request
        assert request.headers["Content-Type"] == "application/x-ndjson"

    @respx.mock
    def test_upsert_records_ndjson_body_format(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(201),
        )
        idx = _make_index()
        idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}, {"_id": "r2", "text": "world"}],
        )

        body = route.calls.last.request.content.decode("utf-8")
        lines = body.strip().split("\n")
        assert len(lines) == 2
        # Each line should be valid JSON
        parsed_0 = json.loads(lines[0])
        parsed_1 = json.loads(lines[1])
        assert parsed_0["_id"] == "r1"
        assert parsed_1["_id"] == "r2"

    @respx.mock
    def test_upsert_records_id_alias(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(201),
        )
        idx = _make_index()
        idx.upsert_records(
            namespace="test-ns",
            records=[{"id": "r1", "text": "hello"}],
        )

        body = route.calls.last.request.content.decode("utf-8")
        parsed = json.loads(body.strip())
        assert "_id" in parsed
        assert "id" not in parsed
        assert parsed["_id"] == "r1"

    def test_upsert_records_empty_list(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="non-empty"):
            idx.upsert_records(namespace="test-ns", records=[])

    def test_upsert_records_missing_id(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="'_id' or 'id'"):
            idx.upsert_records(
                namespace="test-ns",
                records=[{"text": "no id"}],
            )

    @respx.mock
    def test_upsert_records_mixed_id_formats(self) -> None:
        route = respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(201),
        )
        idx = _make_index()
        result = idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "a"}, {"id": "r2", "text": "b"}],
        )

        assert result.record_count == 2
        body = route.calls.last.request.content.decode("utf-8")
        lines = body.strip().split("\n")
        parsed_0 = json.loads(lines[0])
        parsed_1 = json.loads(lines[1])
        # First record keeps _id as-is
        assert parsed_0["_id"] == "r1"
        # Second record has id normalized to _id
        assert parsed_1["_id"] == "r2"
        assert "id" not in parsed_1

    def test_upsert_records_keyword_only(self) -> None:
        idx = _make_index()
        with pytest.raises(TypeError):
            idx.upsert_records([{"_id": "r1"}], "test-ns")  # type: ignore[misc]

    @respx.mock
    def test_upsert_records_response_bracket_access(self) -> None:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(201),
        )
        idx = _make_index()
        result = idx.upsert_records(
            namespace="test-ns",
            records=[{"_id": "r1", "text": "hello"}],
        )

        assert result["record_count"] == 1
