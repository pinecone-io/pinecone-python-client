"""Unit tests for PreviewIndexes.list()."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import PineconeValueError
from pinecone.models.pagination import Page, Paginator
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.indexes import PreviewIndexes
from pinecone.preview.models.indexes import PreviewIndexModel

BASE_URL = "https://api.test.pinecone.io"

_INDEX_1: dict = {
    "name": "index-one",
    "host": "index-one-xyz.svc.pinecone.io",
    "status": {"ready": True, "state": "Ready"},
    "schema": {"fields": {"e": {"type": "dense_vector", "dimension": 4}}},
    "deployment": {
        "deployment_type": "managed",
        "environment": "us-east-1-aws",
        "cloud": "aws",
        "region": "us-east-1",
    },
    "deletion_protection": "disabled",
}

_INDEX_2: dict = {
    "name": "index-two",
    "host": "index-two-xyz.svc.pinecone.io",
    "status": {"ready": False, "state": "Initializing"},
    "schema": {"fields": {"v": {"type": "dense_vector", "dimension": 8}}},
    "deployment": {
        "deployment_type": "managed",
        "environment": "us-east-1-aws",
        "cloud": "aws",
        "region": "us-east-1",
    },
    "deletion_protection": "disabled",
}


@pytest.fixture
def indexes() -> PreviewIndexes:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return PreviewIndexes(config=config)


@respx.mock
def test_list_returns_paginator_yielding_models(indexes: PreviewIndexes) -> None:
    """list() yields PreviewIndexModel instances for each index in the response."""
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": [_INDEX_1, _INDEX_2]})
    )

    result = indexes.list()
    assert isinstance(result, Paginator)

    items = list(result)
    assert len(items) == 2
    assert all(isinstance(item, PreviewIndexModel) for item in items)
    assert items[0].name == "index-one"
    assert items[1].name == "index-two"


@respx.mock
def test_list_sends_api_version_header(indexes: PreviewIndexes) -> None:
    """list() carries the X-Pinecone-Api-Version: 2026-01.alpha header."""
    route = respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": [_INDEX_1]})
    )

    list(indexes.list())

    assert route.called
    request = route.calls.last.request
    assert request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@respx.mock
def test_list_empty_response(indexes: PreviewIndexes) -> None:
    """list() on an empty index set yields zero items; to_list() returns []."""
    respx.get(f"{BASE_URL}/indexes").mock(return_value=httpx.Response(200, json={"indexes": []}))

    items = list(indexes.list())
    assert items == []

    respx.get(f"{BASE_URL}/indexes").mock(return_value=httpx.Response(200, json={"indexes": []}))
    assert indexes.list().to_list() == []


@respx.mock
def test_list_respects_limit(indexes: PreviewIndexes) -> None:
    """list(limit=2) yields at most 2 items even when the response has more."""
    five_indexes = [
        {
            "name": f"index-{i}",
            "host": f"index-{i}.svc.pinecone.io",
            "status": {"ready": True, "state": "Ready"},
            "schema": {"fields": {"e": {"type": "dense_vector", "dimension": 4}}},
            "deployment": {
                "deployment_type": "managed",
                "environment": "us-east-1-aws",
                "cloud": "aws",
                "region": "us-east-1",
            },
            "deletion_protection": "disabled",
        }
        for i in range(5)
    ]
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": five_indexes})
    )

    items = list(indexes.list(limit=2))
    assert len(items) == 2


@respx.mock
def test_list_empty_response_pages_terminates(indexes: PreviewIndexes) -> None:
    """list().pages() yields exactly one Page with empty items and then terminates."""
    respx.get(f"{BASE_URL}/indexes").mock(return_value=httpx.Response(200, json={"indexes": []}))

    pages = list(indexes.list().pages())
    assert len(pages) == 1
    assert isinstance(pages[0], Page)
    assert pages[0].items == []


@respx.mock
def test_list_to_list_collects_all(indexes: PreviewIndexes) -> None:
    """list().to_list() returns all items as a flat list."""
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": [_INDEX_1, _INDEX_2, _INDEX_1]})
    )

    result = indexes.list().to_list()
    assert len(result) == 3


def test_list_rejects_non_positive_limit(indexes: PreviewIndexes) -> None:
    """list(limit=0) and list(limit=-1) raise PineconeValueError without HTTP calls."""
    with pytest.raises(PineconeValueError):
        indexes.list(limit=0)

    with pytest.raises(PineconeValueError):
        indexes.list(limit=-1)
