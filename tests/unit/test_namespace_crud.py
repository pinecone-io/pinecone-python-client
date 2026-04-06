"""Unit tests for Index.create_namespace, describe_namespace, delete_namespace."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone import Index
from pinecone.errors.exceptions import ValidationError
from pinecone.models.namespaces.models import NamespaceDescription

INDEX_HOST = "my-index-abc123.svc.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
NS_URL = f"{INDEX_HOST_HTTPS}/namespaces"


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# create_namespace
# ---------------------------------------------------------------------------


class TestCreateNamespace:
    """Create namespace (unified-ns-0001, unified-ns-0002)."""

    @respx.mock
    def test_create_namespace_basic(self) -> None:
        respx.post(NS_URL).mock(
            return_value=httpx.Response(200, json={"name": "ns1", "record_count": 0}),
        )
        idx = _make_index()
        result = idx.create_namespace(name="ns1")

        assert isinstance(result, NamespaceDescription)
        assert result.name == "ns1"
        assert result.record_count == 0

    @respx.mock
    def test_create_namespace_with_schema(self) -> None:
        respx.post(NS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "ns1",
                    "record_count": 0,
                    "schema": {"fields": {"genre": {"filterable": True}}},
                },
            ),
        )
        idx = _make_index()
        result = idx.create_namespace(
            name="ns1",
            schema={"fields": {"genre": {"filterable": True}}},
        )

        assert result.schema is not None
        assert "genre" in result.schema.fields
        assert result.schema.fields["genre"].filterable is True

    @respx.mock
    def test_create_namespace_request_body(self) -> None:
        route = respx.post(NS_URL).mock(
            return_value=httpx.Response(200, json={"name": "ns1", "record_count": 0}),
        )
        idx = _make_index()
        idx.create_namespace(
            name="ns1",
            schema={"fields": {"year": {"filterable": True}}},
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["name"] == "ns1"
        assert body["schema"] == {"fields": {"year": {"filterable": True}}}

    def test_create_namespace_empty_name(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="non-empty string"):
            idx.create_namespace(name="")

    def test_create_namespace_whitespace_name(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="non-empty string"):
            idx.create_namespace(name="  ")

    def test_create_namespace_non_string_name(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="must be a string"):
            idx.create_namespace(name=123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# describe_namespace
# ---------------------------------------------------------------------------


class TestDescribeNamespace:
    """Describe namespace (unified-ns-0003)."""

    @respx.mock
    def test_describe_namespace(self) -> None:
        respx.get(f"{NS_URL}/ns1").mock(
            return_value=httpx.Response(200, json={"name": "ns1", "record_count": 500}),
        )
        idx = _make_index()
        result = idx.describe_namespace(name="ns1")

        assert isinstance(result, NamespaceDescription)
        assert result.name == "ns1"
        assert result.record_count == 500

    @respx.mock
    def test_describe_namespace_with_schema(self) -> None:
        respx.get(f"{NS_URL}/ns1").mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "ns1",
                    "record_count": 500,
                    "schema": {"fields": {"genre": {"filterable": True}}},
                    "indexed_fields": {"fields": ["genre"]},
                },
            ),
        )
        idx = _make_index()
        result = idx.describe_namespace(name="ns1")

        assert result.name == "ns1"
        assert result.record_count == 500
        assert result.schema is not None
        assert "genre" in result.schema.fields
        assert result.indexed_fields is not None
        assert "genre" in result.indexed_fields.fields

    def test_describe_namespace_empty_name(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="non-empty string"):
            idx.describe_namespace(name="")


# ---------------------------------------------------------------------------
# delete_namespace
# ---------------------------------------------------------------------------


class TestDeleteNamespace:
    """Delete namespace (unified-ns-0004)."""

    @respx.mock
    def test_delete_namespace(self) -> None:
        route = respx.delete(f"{NS_URL}/ns1").mock(
            return_value=httpx.Response(200, json={}),
        )
        idx = _make_index()
        result = idx.delete_namespace(name="ns1")

        assert result is None
        assert route.called

    def test_delete_namespace_empty_name(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="non-empty string"):
            idx.delete_namespace(name="")


# ---------------------------------------------------------------------------
# Keyword-only enforcement
# ---------------------------------------------------------------------------


class TestKeywordOnly:
    """All namespace CRUD methods require keyword arguments."""

    def test_create_namespace_keyword_only(self) -> None:
        idx = _make_index()
        with pytest.raises(TypeError):
            idx.create_namespace("ns1")  # type: ignore[misc]

    def test_describe_namespace_keyword_only(self) -> None:
        idx = _make_index()
        with pytest.raises(TypeError):
            idx.describe_namespace("ns1")  # type: ignore[misc]

    def test_delete_namespace_keyword_only(self) -> None:
        idx = _make_index()
        with pytest.raises(TypeError):
            idx.delete_namespace("ns1")  # type: ignore[misc]
