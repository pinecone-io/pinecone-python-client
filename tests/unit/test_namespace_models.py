"""Unit tests for namespace response models and adapter decoding."""

from __future__ import annotations

import pytest

from pinecone._internal.adapters.vectors_adapter import VectorsAdapter
from pinecone.models.namespaces.models import (
    IndexedFields,
    ListNamespacesResponse,
    NamespaceDescription,
    NamespaceFieldConfig,
    NamespaceSchema,
)
from pinecone.models.vectors.responses import Pagination


class TestNamespaceDescription:
    def test_namespace_description_basic(self) -> None:
        ns = NamespaceDescription(name="ns1", record_count=100)
        assert ns.name == "ns1"
        assert ns.record_count == 100

    def test_namespace_description_with_schema(self) -> None:
        ns = NamespaceDescription(
            name="ns1",
            record_count=50,
            schema=NamespaceSchema(fields={"genre": NamespaceFieldConfig(filterable=True)}),
        )
        assert ns.schema is not None
        assert ns.schema.fields["genre"].filterable is True

    def test_namespace_description_bracket_access(self) -> None:
        ns = NamespaceDescription(name="ns1", record_count=100)
        assert ns["name"] == "ns1"
        assert ns["record_count"] == 100
        with pytest.raises(KeyError):
            ns["missing"]

    def test_namespace_description_defaults(self) -> None:
        ns = NamespaceDescription()
        assert ns.name == ""
        assert ns.record_count == 0
        assert ns.schema is None
        assert ns.indexed_fields is None

    def test_namespace_description_with_indexed_fields(self) -> None:
        ns = NamespaceDescription(
            name="ns1",
            indexed_fields=IndexedFields(fields=["genre", "year"]),
        )
        assert ns.indexed_fields is not None
        assert ns.indexed_fields.fields == ["genre", "year"]


class TestListNamespacesResponse:
    def test_list_namespaces_response_with_pagination(self) -> None:
        response = ListNamespacesResponse(
            namespaces=[NamespaceDescription(name="ns1", record_count=10)],
            pagination=Pagination(next="token123"),
            total_count=1,
        )
        assert response.pagination is not None
        assert response.pagination.next == "token123"

    def test_list_namespaces_response_last_page(self) -> None:
        response = ListNamespacesResponse(
            namespaces=[NamespaceDescription(name="ns1")],
            pagination=None,
            total_count=1,
        )
        assert response.pagination is None

    def test_list_namespaces_response_total_count(self) -> None:
        response = ListNamespacesResponse(total_count=25)
        assert response.total_count == 25

    def test_list_namespaces_response_bracket_access(self) -> None:
        response = ListNamespacesResponse(total_count=5)
        assert response["total_count"] == 5
        assert response["namespaces"] == []
        with pytest.raises(KeyError):
            response["missing"]


class TestVectorsAdapterNamespaces:
    def test_adapter_decode_namespace_description(self) -> None:
        data = (
            b'{"name": "ns1", "record_count": 500,'
            b' "schema": {"fields": {"genre": {"filterable": true}}}}'
        )
        ns = VectorsAdapter.to_namespace_description(data)
        assert ns.name == "ns1"
        assert ns.record_count == 500
        assert ns.schema is not None
        assert ns.schema.fields["genre"].filterable is True

    def test_adapter_decode_list_namespaces(self) -> None:
        data = (
            b'{"namespaces": [{"name": "ns1", "record_count": 100},'
            b' {"name": "ns2", "record_count": 200}],'
            b' "pagination": {"next": "tok"}, "total_count": 2}'
        )
        response = VectorsAdapter.to_list_namespaces_response(data)
        assert len(response.namespaces) == 2
        assert response.namespaces[0].name == "ns1"
        assert response.namespaces[0].record_count == 100
        assert response.namespaces[1].name == "ns2"
        assert response.namespaces[1].record_count == 200
        assert response.pagination is not None
        assert response.pagination.next == "tok"
        assert response.total_count == 2

    def test_adapter_decode_list_namespaces_no_pagination(self) -> None:
        data = b'{"namespaces": [{"name": "ns1", "record_count": 10}], "total_count": 1}'
        response = VectorsAdapter.to_list_namespaces_response(data)
        assert response.pagination is None
        assert len(response.namespaces) == 1
