"""Unit tests for to_dict() on namespace models."""

from __future__ import annotations

from msgspec import Struct

from pinecone.models.namespaces.models import (
    IndexedFields,
    ListNamespacesResponse,
    NamespaceDescription,
    NamespaceFieldConfig,
    NamespaceSchema,
)


def test_namespace_field_config_to_dict() -> None:
    result = NamespaceFieldConfig(filterable=True).to_dict()
    assert result == {"filterable": True}


def test_namespace_schema_to_dict_empty_fields() -> None:
    result = NamespaceSchema().to_dict()
    assert result == {"fields": {}}


def test_namespace_schema_to_dict_nested_field_config() -> None:
    schema = NamespaceSchema(fields={"genre": NamespaceFieldConfig(filterable=True)})
    result = schema.to_dict()
    assert isinstance(result["fields"]["genre"], dict)
    assert result["fields"]["genre"] == {"filterable": True}
    assert not isinstance(result["fields"]["genre"], Struct)


def test_indexed_fields_to_dict() -> None:
    result = IndexedFields(fields=["genre", "year"]).to_dict()
    assert result == {"fields": ["genre", "year"]}


def test_namespace_description_to_dict_required_only() -> None:
    result = NamespaceDescription(name="ns1", record_count=100).to_dict()
    assert "name" in result
    assert "record_count" in result
    assert "schema" in result
    assert "indexed_fields" in result
    assert result["name"] == "ns1"
    assert result["record_count"] == 100
    assert result["schema"] is None
    assert result["indexed_fields"] is None


def test_namespace_description_to_dict_nested_schema() -> None:
    schema = NamespaceSchema(fields={"genre": NamespaceFieldConfig(filterable=True)})
    ns = NamespaceDescription(name="ns1", record_count=10, schema=schema)
    result = ns.to_dict()
    assert isinstance(result["schema"], dict)
    assert not isinstance(result["schema"], Struct)
    assert result["schema"]["fields"]["genre"] == {"filterable": True}


def test_list_namespaces_response_to_dict_empty() -> None:
    result = ListNamespacesResponse().to_dict()
    assert result == {"namespaces": [], "pagination": None, "total_count": 0}


def test_list_namespaces_response_to_dict_nested() -> None:
    ns = NamespaceDescription(name="ns1", record_count=42)
    response = ListNamespacesResponse(namespaces=[ns], total_count=1)
    result = response.to_dict()
    assert isinstance(result["namespaces"], list)
    assert len(result["namespaces"]) == 1
    assert isinstance(result["namespaces"][0], dict)
    assert not isinstance(result["namespaces"][0], Struct)
    assert result["namespaces"][0]["name"] == "ns1"
    assert result["namespaces"][0]["record_count"] == 42


def test_to_dict_is_pure_read_namespace_description() -> None:
    ns = NamespaceDescription(name="ns1", record_count=5)
    result = ns.to_dict()
    result["name"] = "mutated"
    assert ns.name == "ns1"


def test_to_dict_is_pure_read_list_namespaces_response() -> None:
    ns = NamespaceDescription(name="ns1", record_count=5)
    response = ListNamespacesResponse(namespaces=[ns], total_count=1)
    result = response.to_dict()
    result["namespaces"].clear()
    assert len(response.namespaces) == 1
