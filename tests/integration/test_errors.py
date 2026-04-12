"""Integration tests for error paths (sync / REST + gRPC).

Tests verify that the SDK raises typed, human-readable exceptions rather than
raw HTTP errors or generic exceptions.
"""

from __future__ import annotations

import pytest

from pinecone import GrpcIndex, Index, Pinecone, PineconeValueError
from pinecone.errors import ApiError, ConflictError, NotFoundError, UnauthorizedError
from pinecone.models.indexes.specs import ServerlessSpec
from tests.integration.conftest import cleanup_resource, unique_name

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


# ---------------------------------------------------------------------------
# error-dimension-mismatch
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_dimension_mismatch_raises_typed_error_rest(client: Pinecone) -> None:
    """Upsert a 3-dim vector into a 2-dim index raises ApiError (status_code=400, REST sync)."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        with pytest.raises(ApiError) as exc_info:
            index.upsert(
                vectors=[{"id": "dim-v1", "values": [0.1, 0.2, 0.3]}]
            )

        err = exc_info.value
        assert err.status_code == 400
        # Error message must be human-readable
        msg = str(err)
        assert len(msg) > 0
        assert not msg.strip().isdigit()
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


@pytest.mark.integration
def test_dimension_mismatch_raises_typed_error_grpc(client: Pinecone) -> None:
    """Upsert a 3-dim vector into a 2-dim index raises ApiError (status_code=400, gRPC)."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

        with pytest.raises(ApiError) as exc_info:
            index.upsert(
                vectors=[{"id": "dim-v1", "values": [0.1, 0.2, 0.3]}]
            )

        err = exc_info.value
        assert err.status_code == 400
        msg = str(err)
        assert len(msg) > 0
        assert not msg.strip().isdigit()
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# error-duplicate-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_duplicate_index_raises_conflict_error(client: Pinecone) -> None:
    """Creating an index with a name that already exists raises ConflictError (status_code=409)."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        with pytest.raises(ConflictError) as exc_info:
            client.indexes.create(
                name=name,
                dimension=2,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                timeout=-1,  # skip waiting — index already exists
            )

        err = exc_info.value
        assert isinstance(err, ApiError)
        assert err.status_code == 409
        msg = str(err)
        assert len(msg) > 0
        assert not msg.strip().isdigit()
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# error-invalid-host  (unified-index-0043)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_invalid_index_host_raises_value_error() -> None:
    """Index and GrpcIndex raise PineconeValueError for hosts without a dot or 'localhost'.

    Verifies unified-index-0043: host URL validation fires at construction time,
    before any network call is attempted. A host string must contain a dot or
    the substring 'localhost' to be considered a plausible URL.
    """
    # REST Index: no-dot host rejected
    with pytest.raises(PineconeValueError):
        Index(host="nodot", api_key="testkey")

    # REST Index: empty string rejected
    with pytest.raises(PineconeValueError):
        Index(host="", api_key="testkey")

    # REST Index: whitespace-only rejected
    with pytest.raises(PineconeValueError):
        Index(host="   ", api_key="testkey")

    # GrpcIndex: same validation applies
    with pytest.raises(PineconeValueError):
        GrpcIndex(host="nodot", api_key="testkey")

    with pytest.raises(PineconeValueError):
        GrpcIndex(host="", api_key="testkey")
