"""Integration tests for error paths (sync / REST + gRPC).

Tests verify that the SDK raises typed, human-readable exceptions rather than
raw HTTP errors or generic exceptions.
"""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.errors import ApiError, NotFoundError, UnauthorizedError


# ---------------------------------------------------------------------------
# error-bad-api-key
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_bad_api_key_raises_typed_exception() -> None:
    """Pinecone(api_key="invalid") + indexes.list() raises UnauthorizedError (not raw HTTP error)."""
    bad_client = Pinecone(api_key="invalid-key-12345")
    with pytest.raises(UnauthorizedError) as exc_info:
        bad_client.indexes.list()

    err = exc_info.value
    assert isinstance(err, ApiError)
    assert err.status_code == 401
    # Error message must be human-readable (non-empty)
    assert str(err)


@pytest.mark.integration
def test_bad_api_key_error_message_is_human_readable() -> None:
    """UnauthorizedError from a bad API key has a non-empty, informative message."""
    bad_client = Pinecone(api_key="totally-wrong-key-xyz")
    with pytest.raises(UnauthorizedError) as exc_info:
        bad_client.indexes.list()

    err = exc_info.value
    # Message should exist and not just be a raw status code
    msg = str(err)
    assert len(msg) > 0
    # Should not be only a number
    assert not msg.strip().isdigit()


# ---------------------------------------------------------------------------
# error-nonexistent-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_describe_nonexistent_index_raises_not_found(client: Pinecone) -> None:
    """indexes.describe() on a non-existent name raises NotFoundError (typed, status_code=404)."""
    with pytest.raises(NotFoundError) as exc_info:
        client.indexes.describe("index-that-does-not-exist-xyz")

    err = exc_info.value
    assert isinstance(err, ApiError)
    assert err.status_code == 404
    # Error message must be human-readable (non-empty, not just a number)
    msg = str(err)
    assert len(msg) > 0
    assert not msg.strip().isdigit()


@pytest.mark.integration
def test_delete_nonexistent_index_raises_not_found(client: Pinecone) -> None:
    """indexes.delete() on a non-existent name raises NotFoundError (typed, status_code=404)."""
    with pytest.raises(NotFoundError) as exc_info:
        client.indexes.delete("index-that-does-not-exist-xyz")

    err = exc_info.value
    assert isinstance(err, ApiError)
    assert err.status_code == 404
