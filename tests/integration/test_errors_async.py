"""Integration tests for error paths (async / REST async).

Tests verify that the async SDK raises typed, human-readable exceptions rather
than raw HTTP errors or generic exceptions.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from pinecone import AsyncPinecone
from pinecone.errors import ApiError, NotFoundError, UnauthorizedError
from pinecone.models.indexes.specs import ServerlessSpec
from tests.integration.conftest import async_cleanup_resource, unique_name


# ---------------------------------------------------------------------------
# error-bad-api-key
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bad_api_key_raises_typed_exception_async() -> None:
    """AsyncPinecone(api_key="invalid") + indexes.list() raises UnauthorizedError (not raw HTTP error)."""
    async with AsyncPinecone(api_key="invalid-key-12345") as bad_client:
        with pytest.raises(UnauthorizedError) as exc_info:
            await bad_client.indexes.list()

    err = exc_info.value
    assert isinstance(err, ApiError)
    assert err.status_code == 401
    # Error message must be human-readable (non-empty)
    assert str(err)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bad_api_key_error_message_is_human_readable_async() -> None:
    """UnauthorizedError from a bad API key has a non-empty, informative message."""
    async with AsyncPinecone(api_key="totally-wrong-key-xyz") as bad_client:
        with pytest.raises(UnauthorizedError) as exc_info:
            await bad_client.indexes.list()

    err = exc_info.value
    msg = str(err)
    assert len(msg) > 0
    assert not msg.strip().isdigit()


# ---------------------------------------------------------------------------
# error-nonexistent-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_describe_nonexistent_index_raises_not_found_async(
    async_client: AsyncPinecone,
) -> None:
    """indexes.describe() on a non-existent name raises NotFoundError (typed, status_code=404)."""
    with pytest.raises(NotFoundError) as exc_info:
        await async_client.indexes.describe("index-that-does-not-exist-xyz")

    err = exc_info.value
    assert isinstance(err, ApiError)
    assert err.status_code == 404
    # Error message must be human-readable (non-empty, not just a number)
    msg = str(err)
    assert len(msg) > 0
    assert not msg.strip().isdigit()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_nonexistent_index_raises_not_found_async(
    async_client: AsyncPinecone,
) -> None:
    """indexes.delete() on a non-existent name raises NotFoundError (typed, status_code=404)."""
    with pytest.raises(NotFoundError) as exc_info:
        await async_client.indexes.delete("index-that-does-not-exist-xyz")

    err = exc_info.value
    assert isinstance(err, ApiError)
    assert err.status_code == 404


# ---------------------------------------------------------------------------
# error-dimension-mismatch
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dimension_mismatch_raises_typed_error_async(
    async_client: AsyncPinecone,
) -> None:
    """Upsert a 3-dim vector into a 2-dim index raises ApiError (status_code=400, REST async)."""
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        # Populate host cache so pc.index(name=...) can resolve it
        await async_client.indexes.describe(name)
        index = async_client.index(name=name)

        with pytest.raises(ApiError) as exc_info:
            await index.upsert(
                vectors=[{"id": "dim-v1", "values": [0.1, 0.2, 0.3]}]
            )

        err = exc_info.value
        assert err.status_code == 400
        msg = str(err)
        assert len(msg) > 0
        assert not msg.strip().isdigit()
    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name), name, "index"
        )
