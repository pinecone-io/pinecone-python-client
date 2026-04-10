"""Integration tests for error paths (async / REST async).

Tests verify that the async SDK raises typed, human-readable exceptions rather
than raw HTTP errors or generic exceptions.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from pinecone import AsyncPinecone
from pinecone.errors import ApiError, UnauthorizedError


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
