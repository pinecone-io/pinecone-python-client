"""Unit tests for PreviewIndexes.delete()."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import ForbiddenError, PineconeTimeoutError, PineconeValueError
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.indexes import PreviewIndexes

BASE_URL = "https://api.test.pinecone.io"

_PREVIEW_INDEX_RESPONSE: dict = {
    "name": "x",
    "host": "x-xyz.svc.pinecone.io",
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

_NOT_FOUND_RESPONSE: dict = {
    "error": {"code": "NOT_FOUND", "message": "Index 'x' not found."},
    "status": 404,
}

_FORBIDDEN_RESPONSE: dict = {
    "error": {"code": "FORBIDDEN", "message": "Deletion protection is enabled."},
    "status": 403,
}


@pytest.fixture
def indexes() -> PreviewIndexes:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return PreviewIndexes(config=config)


@respx.mock
def test_delete_returns_immediately_when_timeout_is_negative_one(indexes: PreviewIndexes) -> None:
    """delete("x", timeout=-1) returns without polling after the DELETE."""
    respx.delete(f"{BASE_URL}/indexes/x").mock(return_value=httpx.Response(204))

    with patch("time.sleep", side_effect=AssertionError("sleep must not be called")):
        with patch("time.monotonic", side_effect=AssertionError("monotonic must not be called")):
            indexes.delete("x", timeout=-1)


@respx.mock
def test_delete_polls_until_not_found(indexes: PreviewIndexes) -> None:
    """delete() polls describe() until it raises NotFoundError."""
    respx.delete(f"{BASE_URL}/indexes/x").mock(return_value=httpx.Response(204))

    describe_route = respx.get(f"{BASE_URL}/indexes/x")
    describe_route.side_effect = [
        httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE),
        httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE),
        httpx.Response(404, json=_NOT_FOUND_RESPONSE),
    ]

    with patch("time.sleep"):
        indexes.delete("x")

    assert describe_route.call_count == 3


@respx.mock
def test_delete_raises_timeout_error_after_deadline(indexes: PreviewIndexes) -> None:
    """delete() raises PineconeTimeoutError when timeout expires before index is gone."""
    respx.delete(f"{BASE_URL}/indexes/x").mock(return_value=httpx.Response(204))
    respx.get(f"{BASE_URL}/indexes/x").mock(
        return_value=httpx.Response(200, json=_PREVIEW_INDEX_RESPONSE)
    )

    monotonic_values = iter([0.0, 10.0])
    with patch("time.sleep"), patch("time.monotonic", side_effect=monotonic_values):
        with pytest.raises(PineconeTimeoutError, match="5s"):
            indexes.delete("x", timeout=5)


@respx.mock
def test_delete_raises_forbidden_when_protection_enabled(indexes: PreviewIndexes) -> None:
    """delete() raises ForbiddenError when the server returns 403."""
    respx.delete(f"{BASE_URL}/indexes/x").mock(
        return_value=httpx.Response(403, json=_FORBIDDEN_RESPONSE)
    )

    with pytest.raises(ForbiddenError):
        indexes.delete("x", timeout=-1)


def test_delete_raises_value_error_on_empty_name(indexes: PreviewIndexes) -> None:
    """delete("") raises PineconeValueError without issuing an HTTP request."""
    with pytest.raises(PineconeValueError):
        indexes.delete("")


@respx.mock
def test_delete_sends_api_version_header(indexes: PreviewIndexes) -> None:
    """DELETE /indexes/{name} carries the preview api-version header."""
    route = respx.delete(f"{BASE_URL}/indexes/x").mock(return_value=httpx.Response(204))

    indexes.delete("x", timeout=-1)

    assert route.called
    assert route.calls.last.request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION
