"""Unit tests for AsyncIndexes namespace — list, describe, exists, create, delete, configure."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION
from pinecone._internal.http_client import AsyncHTTPClient
from pinecone.async_client.indexes import AsyncIndexes
from pinecone.errors.exceptions import NotFoundError, PineconeError, ValidationError
from pinecone.models.enums import DeletionProtection, Metric, VectorType
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.list import IndexList
from pinecone.models.indexes.specs import PodSpec, ServerlessSpec
from tests.factories import (
    make_error_response,
    make_index_list_response,
    make_index_response,
)

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture()
def async_http_client() -> AsyncHTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)


@pytest.fixture()
def async_indexes(async_http_client: AsyncHTTPClient) -> AsyncIndexes:
    return AsyncIndexes(http=async_http_client)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@respx.mock
async def test_list_indexes(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json=make_index_list_response()),
    )

    result = await async_indexes.list()

    assert isinstance(result, IndexList)
    assert len(result) == 1
    assert result[0].name == "test-index"
    assert result.names() == ["test-index"]

    # verify iteration
    names = [idx.name for idx in result]
    assert names == ["test-index"]


@respx.mock
async def test_list_indexes_empty(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(200, json={"indexes": []}),
    )

    result = await async_indexes.list()

    assert isinstance(result, IndexList)
    assert len(result) == 0
    assert result.names() == []


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@respx.mock
async def test_describe_index(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    result = await async_indexes.describe("test-index")

    assert isinstance(result, IndexModel)
    assert result.name == "test-index"
    assert result.dimension == 1536
    assert result.metric == "cosine"
    assert result.host == "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
    assert result.vector_type == "dense"
    assert result.deletion_protection == "disabled"
    assert result.status.ready is True
    assert result.status.state == "Ready"
    # bracket access
    assert result["name"] == "test-index"
    assert result["dimension"] == 1536


@respx.mock
async def test_describe_index_caches_host(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    await async_indexes.describe("test-index")

    assert "test-index" in async_indexes._host_cache
    expected_host = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
    assert async_indexes._host_cache["test-index"] == expected_host


@respx.mock
async def test_describe_nonexistent_index(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/no-such-index").mock(
        return_value=httpx.Response(
            404,
            json=make_error_response(404, "Index not found"),
        ),
    )

    with pytest.raises(NotFoundError):
        await async_indexes.describe("no-such-index")


async def test_describe_empty_name_raises(async_indexes: AsyncIndexes) -> None:
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.describe("")

    assert "name" in str(exc_info.value)
    assert "non-empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------


@respx.mock
async def test_exists_true(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    assert await async_indexes.exists("test-index") is True


@respx.mock
async def test_exists_false(async_indexes: AsyncIndexes) -> None:
    respx.get(f"{BASE_URL}/indexes/no-such-index").mock(
        return_value=httpx.Response(
            404,
            json=make_error_response(404, "Index not found"),
        ),
    )

    assert await async_indexes.exists("no-such-index") is False


async def test_exists_empty_name_raises(async_indexes: AsyncIndexes) -> None:
    with pytest.raises(ValidationError):
        await async_indexes.exists("")


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@respx.mock
async def test_delete_index_no_polling(async_indexes: AsyncIndexes) -> None:
    """DELETE /indexes/test-index -> 202, returns immediately (no polling)."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202),
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/test-index")

    result = await async_indexes.delete("test-index")

    assert result is None
    assert describe_route.call_count == 0


@respx.mock
async def test_delete_removes_host_cache(async_indexes: AsyncIndexes) -> None:
    """Deleting an index removes its cached host URL."""
    async_indexes._host_cache["my-index"] = "my-index-abc.svc.pinecone.io"

    respx.delete(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(202),
    )

    await async_indexes.delete("my-index")

    assert "my-index" not in async_indexes._host_cache


@respx.mock
async def test_delete_polls_until_gone(async_indexes: AsyncIndexes) -> None:
    """With explicit timeout, poll describe until index disappears."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202),
    )
    describe_route = respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(200, json=make_index_response()),
            httpx.Response(404, json=make_error_response(404, "Not found")),
        ],
    )

    with patch("pinecone.async_client.indexes.asyncio.sleep"):
        await async_indexes.delete("test-index", timeout=300)

    assert describe_route.call_count == 2


@respx.mock
async def test_delete_timeout_exceeded(async_indexes: AsyncIndexes) -> None:
    """If index still exists after timeout, raise PineconeError."""
    respx.delete(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response()),
    )

    with patch("pinecone.async_client.indexes.asyncio.sleep"):
        with pytest.raises(PineconeError, match=r"still exists after 1s"):
            await async_indexes.delete("test-index", timeout=1)


@respx.mock
async def test_delete_nonexistent_index(async_indexes: AsyncIndexes) -> None:
    """DELETE on non-existent index -> 404 -> NotFoundError."""
    respx.delete(f"{BASE_URL}/indexes/no-such-index").mock(
        return_value=httpx.Response(404, json=make_error_response(404, "Index not found")),
    )

    with pytest.raises(NotFoundError):
        await async_indexes.delete("no-such-index")


async def test_delete_empty_name_raises(async_indexes: AsyncIndexes) -> None:
    """Empty name raises ValidationError before any HTTP call."""
    with pytest.raises(ValidationError):
        await async_indexes.delete("")


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


@respx.mock
async def test_create_serverless_index(async_indexes: AsyncIndexes) -> None:
    """Create with ServerlessSpec — verify POST body has correct shape."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    result = await async_indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    assert isinstance(result, IndexModel)
    assert result.name == "test-index"

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["name"] == "test-index"
    assert body["dimension"] == 1536
    assert body["metric"] == "cosine"
    assert body["vector_type"] == "dense"
    assert body["deletion_protection"] == "disabled"
    assert body["spec"] == {"serverless": {"cloud": "aws", "region": "us-east-1"}}


@respx.mock
async def test_create_pod_index(async_indexes: AsyncIndexes) -> None:
    """Create with PodSpec — verify body includes pod spec."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(
                spec={
                    "pod": {
                        "environment": "us-east1-gcp",
                        "pod_type": "p1.x1",
                        "replicas": 1,
                        "shards": 1,
                        "pods": 1,
                    }
                }
            ),
        ),
    )

    result = await async_indexes.create(
        name="test-index",
        dimension=1536,
        spec=PodSpec(environment="us-east1-gcp"),
    )

    assert isinstance(result, IndexModel)

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["spec"]["pod"]["environment"] == "us-east1-gcp"
    assert body["spec"]["pod"]["pod_type"] == "p1.x1"  # default


@respx.mock
async def test_create_index_defaults(async_indexes: AsyncIndexes) -> None:
    """Omit optional params — verify defaults."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    await async_indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["metric"] == "cosine"
    assert body["vector_type"] == "dense"
    assert body["deletion_protection"] == "disabled"
    assert "tags" not in body


@respx.mock
async def test_create_index_with_tags(async_indexes: AsyncIndexes) -> None:
    """Verify tags are included in request body."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    await async_indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        tags={"env": "test", "team": "ml"},
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["tags"] == {"env": "test", "team": "ml"}


@respx.mock
async def test_create_with_dict_spec(async_indexes: AsyncIndexes) -> None:
    """Pass raw dict spec — verify it's sent as-is."""
    raw_spec: dict[str, Any] = {"serverless": {"cloud": "gcp", "region": "us-central1"}}
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    await async_indexes.create(
        name="test-index",
        dimension=1536,
        spec=raw_spec,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["spec"] == raw_spec


async def test_create_missing_name_raises(async_indexes: AsyncIndexes) -> None:
    """Empty name raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    assert "name" in str(exc_info.value)


async def test_create_missing_spec_raises(async_indexes: AsyncIndexes) -> None:
    """No spec (None) raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="test-index",
            dimension=1536,
            spec=None,  # type: ignore[arg-type]
        )
    assert "spec" in str(exc_info.value)


async def test_create_dense_missing_dimension_raises(async_indexes: AsyncIndexes) -> None:
    """Dense index without dimension raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="test-index",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    assert "dimension" in str(exc_info.value)


async def test_create_invalid_metric_raises(async_indexes: AsyncIndexes) -> None:
    """Invalid metric raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="test-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            metric="hamming",
        )
    assert "metric" in str(exc_info.value)


async def test_create_invalid_deletion_protection_raises(async_indexes: AsyncIndexes) -> None:
    """Invalid deletion protection value raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        await async_indexes.create(
            name="test-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="maybe",
        )
    assert "deletion_protection" in str(exc_info.value)


@respx.mock
async def test_create_with_metric_enum(async_indexes: AsyncIndexes) -> None:
    """Accept Metric enum for the metric parameter."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response(metric="euclidean")),
    )

    await async_indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        metric=Metric.EUCLIDEAN,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["metric"] == "euclidean"


@respx.mock
async def test_create_with_vector_type_enum(async_indexes: AsyncIndexes) -> None:
    """Accept VectorType enum for the vector_type parameter."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(vector_type="sparse", dimension=None),
        ),
    )

    await async_indexes.create(
        name="sparse-enum-index",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        vector_type=VectorType.SPARSE,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["vector_type"] == "sparse"


@respx.mock
async def test_create_with_deletion_protection_enum(async_indexes: AsyncIndexes) -> None:
    """Accept DeletionProtection enum for the deletion_protection parameter."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(201, json=make_index_response()),
    )

    await async_indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection=DeletionProtection.ENABLED,
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["deletion_protection"] == "enabled"


@respx.mock
async def test_create_timeout_none_no_polling(async_indexes: AsyncIndexes) -> None:
    """With timeout=None (default), describe is NOT called after create."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(status={"ready": False, "state": "Initializing"}),
        ),
    )

    result = await async_indexes.create(
        name="test-index",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    assert result.status.ready is False
    assert len(respx.calls) == 1  # only the POST


@respx.mock
async def test_create_polls_until_ready(async_indexes: AsyncIndexes) -> None:
    """With timeout, poll describe until ready."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(status={"ready": False, "state": "Initializing"}),
        ),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        side_effect=[
            httpx.Response(
                200,
                json=make_index_response(status={"ready": False, "state": "Initializing"}),
            ),
            httpx.Response(
                200,
                json=make_index_response(status={"ready": True, "state": "Ready"}),
            ),
        ]
    )

    with patch("pinecone.async_client.indexes.asyncio.sleep"):
        result = await async_indexes.create(
            name="test-index",
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

    assert result.status.ready is True


@respx.mock
async def test_create_init_failed_raises(async_indexes: AsyncIndexes) -> None:
    """If index enters InitializationFailed, raise immediately."""
    respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(status={"ready": False, "state": "Initializing"}),
        ),
    )
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(
            200,
            json=make_index_response(status={"ready": False, "state": "InitializationFailed"}),
        ),
    )

    with patch("pinecone.async_client.indexes.asyncio.sleep"):
        with pytest.raises(PineconeError, match="failed to initialize"):
            await async_indexes.create(
                name="test-index",
                dimension=1536,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                timeout=300,
            )


@respx.mock
async def test_create_sparse_without_dimension(async_indexes: AsyncIndexes) -> None:
    """Sparse index can be created without specifying dimension."""
    route = respx.post(f"{BASE_URL}/indexes").mock(
        return_value=httpx.Response(
            201,
            json=make_index_response(vector_type="sparse", dimension=None),
        ),
    )

    result = await async_indexes.create(
        name="sparse-index",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        vector_type="sparse",
    )

    assert isinstance(result, IndexModel)
    request = route.calls.last.request
    body = json.loads(request.content)
    assert body["vector_type"] == "sparse"
    assert "dimension" not in body


# ---------------------------------------------------------------------------
# configure()
# ---------------------------------------------------------------------------


@respx.mock
async def test_configure_replicas_only(async_indexes: AsyncIndexes) -> None:
    """PATCH body has spec.pod.replicas only; returns None."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure("test-index", replicas=4)

    payload = _request_json(route)
    assert payload == {"spec": {"pod": {"replicas": 4}}}


@respx.mock
async def test_configure_pod_type_only(async_indexes: AsyncIndexes) -> None:
    """PATCH body has spec.pod.pod_type only."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure("test-index", pod_type="p1.x2")

    payload = _request_json(route)
    assert payload == {"spec": {"pod": {"pod_type": "p1.x2"}}}


@respx.mock
async def test_configure_deletion_protection(async_indexes: AsyncIndexes) -> None:
    """PATCH body includes deletion_protection."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure("test-index", deletion_protection="enabled")

    payload = _request_json(route)
    assert payload == {"deletion_protection": "enabled"}


@respx.mock
async def test_configure_tags_with_merging(async_indexes: AsyncIndexes) -> None:
    """Tags are merged with existing tags from describe."""
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(
            200,
            json=make_index_response(tags={"existing": "val", "keep": "me"}),
        ),
    )
    patch_route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure(
        "test-index", tags={"new_tag": "hello", "existing": "overwritten"}
    )

    payload = _request_json(patch_route)
    assert payload == {
        "tags": {"existing": "overwritten", "keep": "me", "new_tag": "hello"},
    }


@respx.mock
async def test_configure_tag_removal_via_empty_string(async_indexes: AsyncIndexes) -> None:
    """Setting a tag value to empty string passes through for removal."""
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(
            200,
            json=make_index_response(tags={"remove_me": "old_val", "keep": "val"}),
        ),
    )
    patch_route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure("test-index", tags={"remove_me": ""})

    payload = _request_json(patch_route)
    assert payload["tags"]["remove_me"] == ""
    assert payload["tags"]["keep"] == "val"


@respx.mock
async def test_configure_deletion_protection_enum(async_indexes: AsyncIndexes) -> None:
    """Accept DeletionProtection enum for deletion_protection param."""
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure("test-index", deletion_protection=DeletionProtection.ENABLED)

    payload = _request_json(route)
    assert payload == {"deletion_protection": "enabled"}


@respx.mock
async def test_configure_multiple_fields(async_indexes: AsyncIndexes) -> None:
    """Can set replicas, pod_type, and deletion_protection together."""
    respx.get(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(200, json=make_index_response(tags={})),
    )
    route = respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    await async_indexes.configure(
        "test-index",
        replicas=4,
        pod_type="p1.x2",
        deletion_protection="disabled",
        tags={"env": "prod"},
    )

    payload = _request_json(route)
    assert payload["spec"]["pod"]["replicas"] == 4
    assert payload["spec"]["pod"]["pod_type"] == "p1.x2"
    assert payload["deletion_protection"] == "disabled"
    assert payload["tags"] == {"env": "prod"}


@respx.mock
async def test_configure_returns_none(async_indexes: AsyncIndexes) -> None:
    """configure() always returns None."""
    respx.patch(f"{BASE_URL}/indexes/test-index").mock(
        return_value=httpx.Response(202, json=make_index_response()),
    )

    assert await async_indexes.configure("test-index", replicas=1) is None  # type: ignore[func-returns-value]


async def test_configure_empty_name_raises(async_indexes: AsyncIndexes) -> None:
    """Empty name raises ValidationError before any HTTP call."""
    with pytest.raises(ValidationError):
        await async_indexes.configure("")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request_json(route: respx.Route) -> dict[str, Any]:
    """Extract the JSON body from the last request on a route."""
    import orjson

    request = route.calls.last.request
    return orjson.loads(request.content)  # type: ignore[no-any-return]
