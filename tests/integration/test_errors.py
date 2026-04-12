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
            index.upsert(vectors=[{"id": "dim-v1", "values": [0.1, 0.2, 0.3]}])

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
            index.upsert(vectors=[{"id": "dim-v1", "values": [0.1, 0.2, 0.3]}])

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


# ---------------------------------------------------------------------------
# error-query-validation  (unified-vec-0038, unified-vec-0039, unified-vec-0040)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# error-invalid-spec-dict-key  (unified-index-0044)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_index_invalid_spec_dict_key(client: Pinecone) -> None:
    """indexes.create() with a spec dict missing a recognized key raises PineconeValueError.

    Verifies unified-index-0044: the SDK rejects spec dicts that do not contain
    a 'serverless', 'pod', or 'byoc' key.  Validation fires synchronously before
    any HTTP request is made, so no index resource is created or cleaned up.

    Three cases are checked:
    - empty dict: no key at all
    - dict with an unrecognized key: {"invalid": {...}}
    - dict with a case-wrong key: {"SERVERLESS": {...}} (case-sensitive match)
    """
    # empty spec dict
    with pytest.raises(PineconeValueError):
        client.indexes.create(
            name="test-idx-spec",
            dimension=2,
            spec={},
        )

    # unrecognized key — value doesn't matter, key is what's checked
    with pytest.raises(PineconeValueError):
        client.indexes.create(
            name="test-idx-spec",
            dimension=2,
            spec={"invalid": {"cloud": "aws", "region": "us-east-1"}},
        )

    # case-sensitive: 'SERVERLESS' is not recognized (must be lowercase)
    with pytest.raises(PineconeValueError):
        client.indexes.create(
            name="test-idx-spec",
            dimension=2,
            spec={"SERVERLESS": {"cloud": "aws", "region": "us-east-1"}},
        )


@pytest.mark.integration
def test_query_input_validation_rest() -> None:
    """query() client-side validation raises PineconeValueError before any API call (REST sync).

    Uses a fake host so no real index or network call is required; all checks
    fire synchronously before the HTTP request would be made.

    Verifies:
    - unified-vec-0038: top_k < 1 is rejected
    - unified-vec-0039: both vector and id supplied is rejected
    - unified-vec-0039: neither vector nor id is rejected
    - unified-vec-0040: positional arguments raise TypeError
    """
    index = Index(host="fake-index.svc.pinecone.io", api_key="testkey")

    # unified-vec-0038: top_k=0 rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=0, vector=[0.1, 0.2])

    # unified-vec-0038: negative top_k rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=-5, vector=[0.1, 0.2])

    # unified-vec-0039: both vector and id rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=5, vector=[0.1, 0.2], id="some-id")

    # unified-vec-0039: neither vector nor id rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=5)

    # unified-vec-0040: positional arguments rejected by Python (keyword-only)
    with pytest.raises(TypeError):
        index.query([0.1, 0.2], 5)  # type: ignore[misc]


@pytest.mark.integration
def test_update_input_validation_rest() -> None:
    """update() client-side validation raises PineconeValueError before any API call (REST sync).

    Uses a fake host so no real index or network call is required; all checks
    fire synchronously before the HTTP request would be made.

    Verifies:
    - unified-vec-0042: both id and filter rejected
    - unified-vec-0042: neither id nor filter rejected
    - update() uses keyword-only params (TypeError on positional args)
    """
    index = Index(host="fake-index.svc.pinecone.io", api_key="testkey")

    # unified-vec-0042: both id and filter rejected
    with pytest.raises(PineconeValueError):
        index.update(id="some-id", filter={"genre": {"$eq": "drama"}}, set_metadata={"x": 1})

    # unified-vec-0042: neither id nor filter rejected
    with pytest.raises(PineconeValueError):
        index.update(set_metadata={"x": 1})

    # update() uses keyword-only params — positional call raises TypeError
    with pytest.raises(TypeError):
        index.update("some-id")  # type: ignore[misc]


@pytest.mark.integration
def test_query_input_validation_grpc() -> None:
    """query() client-side validation raises PineconeValueError before any gRPC call.

    All validations fire before self._call_channel() so no real server is needed.

    Verifies:
    - unified-vec-0038: top_k < 1 is rejected
    - unified-vec-0039: both vector and id supplied is rejected
    - unified-vec-0039: neither vector nor id is rejected
    - unified-vec-0040: positional arguments raise TypeError
    """
    index = GrpcIndex(host="fake-index.svc.pinecone.io", api_key="testkey")

    # unified-vec-0038: top_k=0 rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=0, vector=[0.1, 0.2])

    # unified-vec-0038: negative top_k rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=-3, vector=[0.1, 0.2])

    # unified-vec-0039: both vector and id rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=5, vector=[0.1, 0.2], id="some-id")

    # unified-vec-0039: neither vector nor id rejected
    with pytest.raises(PineconeValueError):
        index.query(top_k=5)

    # unified-vec-0040: positional arguments rejected by Python (keyword-only)
    with pytest.raises(TypeError):
        index.query([0.1, 0.2], 5)  # type: ignore[misc]


@pytest.mark.integration
def test_update_input_validation_grpc() -> None:
    """update() client-side validation raises PineconeValueError before any gRPC call.

    Uses a fake host so no real server is needed; validation fires before
    the gRPC channel is called.

    Verifies:
    - unified-vec-0042: both id and filter rejected
    - unified-vec-0042: neither id nor filter rejected
    """
    index = GrpcIndex(host="fake-index.svc.pinecone.io", api_key="testkey")

    # unified-vec-0042: both id and filter rejected
    with pytest.raises(PineconeValueError):
        index.update(id="some-id", filter={"genre": {"$eq": "drama"}}, set_metadata={"x": 1})

    # unified-vec-0042: neither id nor filter rejected
    with pytest.raises(PineconeValueError):
        index.update(set_metadata={"x": 1})


# ---------------------------------------------------------------------------
# namespace-name-must-be-string — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_namespace_name_must_be_string_rest() -> None:
    """create_namespace(), describe_namespace(), and delete_namespace() raise
    PineconeValueError when the name parameter is not a string.

    Validation fires client-side before any HTTP request, so a fake host is
    sufficient — no real index or API call is needed.

    Verifies:
    - unified-ns-0011: Namespace operations require the namespace parameter to be a string.
    """
    # Fake host: contains a dot so it passes the host URL format check.
    index = Index(host="fake-index.svc.pinecone.io", api_key="testkey")

    non_string_values = [42, None, ["my-ns"], True]

    for bad_name in non_string_values:
        # create_namespace rejects non-string name
        with pytest.raises(PineconeValueError, match="string"):
            index.create_namespace(name=bad_name)  # type: ignore[arg-type]

        # describe_namespace rejects non-string name
        with pytest.raises(PineconeValueError, match="string"):
            index.describe_namespace(name=bad_name)  # type: ignore[arg-type]

        # delete_namespace rejects non-string name
        with pytest.raises(PineconeValueError, match="string"):
            index.delete_namespace(name=bad_name)  # type: ignore[arg-type]
