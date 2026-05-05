"""Unit tests for backup description default — absent from wire payload when not provided."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
from pinecone.async_client.backups import AsyncBackups
from pinecone.client.backups import Backups
from pinecone.models.backups.model import BackupModel
from tests.factories import make_backup_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture
def http_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture
def backups(http_client: HTTPClient) -> Backups:
    return Backups(http=http_client)


@pytest.fixture
async def async_http_client() -> AsyncGenerator[AsyncHTTPClient]:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    client = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
    yield client
    await client.close()


@pytest.fixture
def async_backups(async_http_client: AsyncHTTPClient) -> AsyncBackups:
    return AsyncBackups(http=async_http_client)


@respx.mock
def test_create_backup_description_default(backups: Backups) -> None:
    """When description is not provided, it is absent from the wire payload."""
    route = respx.post(f"{BASE_URL}/indexes/test/backups").mock(
        return_value=httpx.Response(201, json=make_backup_response()),
    )

    result = backups.create(index_name="test")

    assert isinstance(result, BackupModel)
    request = route.calls[0].request
    body = json.loads(request.content)
    assert "description" not in body


@respx.mock
def test_create_backup_explicit_description_sent(backups: Backups) -> None:
    """When description is explicitly provided, it is included in the wire payload."""
    route = respx.post(f"{BASE_URL}/indexes/test/backups").mock(
        return_value=httpx.Response(201, json=make_backup_response()),
    )

    result = backups.create(index_name="test", description="my description")

    assert isinstance(result, BackupModel)
    request = route.calls[0].request
    body = json.loads(request.content)
    assert body["description"] == "my description"


@respx.mock
def test_create_backup_empty_string_description_sent(backups: Backups) -> None:
    """When description is explicitly set to empty string, it is sent in the payload."""
    route = respx.post(f"{BASE_URL}/indexes/test/backups").mock(
        return_value=httpx.Response(201, json=make_backup_response()),
    )

    result = backups.create(index_name="test", description="")

    assert isinstance(result, BackupModel)
    request = route.calls[0].request
    body = json.loads(request.content)
    assert "description" in body
    assert body["description"] == ""


@respx.mock
@pytest.mark.anyio
async def test_async_create_backup_description_default(
    async_backups: AsyncBackups,
) -> None:
    """Async variant: description absent from wire payload when not provided."""
    route = respx.post(f"{BASE_URL}/indexes/test/backups").mock(
        return_value=httpx.Response(201, json=make_backup_response()),
    )

    result = await async_backups.create(index_name="test")

    assert isinstance(result, BackupModel)
    request = route.calls[0].request
    body = json.loads(request.content)
    assert "description" not in body


@respx.mock
@pytest.mark.anyio
async def test_async_create_backup_explicit_description_sent(
    async_backups: AsyncBackups,
) -> None:
    """Async variant: explicit description is included in the wire payload."""
    route = respx.post(f"{BASE_URL}/indexes/test/backups").mock(
        return_value=httpx.Response(201, json=make_backup_response()),
    )

    result = await async_backups.create(index_name="test", description="async desc")

    assert isinstance(result, BackupModel)
    request = route.calls[0].request
    body = json.loads(request.content)
    assert body["description"] == "async desc"
