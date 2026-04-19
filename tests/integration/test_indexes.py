"""Integration tests for index CRUD operations (sync / REST + gRPC)."""

from __future__ import annotations

import pytest

from pinecone import GrpcIndex, Index, Pinecone
from pinecone.errors import ForbiddenError, NotFoundError
from pinecone.models.indexes.index import IndexModel, IndexSpec, IndexStatus
from pinecone.models.indexes.specs import EmbedConfig, IntegratedSpec, ServerlessSpec
from tests.integration.conftest import cleanup_resource, unique_name

# ---------------------------------------------------------------------------
# list-indexes
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_indexes_returns_index_list(client: Pinecone) -> None:
    """pc.indexes.list() returns an IndexList that is iterable and supports len()."""
    result = client.indexes.list()

    # IndexList supports len()
    count = len(result)
    assert isinstance(count, int)
    assert count >= 0

    # IndexList supports iteration
    items = list(result)
    assert len(items) == count

    # .names() returns a list of strings
    names = result.names()
    assert isinstance(names, list)
    assert len(names) == count
    for name in names:
        assert isinstance(name, str)
        assert len(name) > 0


# ---------------------------------------------------------------------------
# create-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_serverless_index_becomes_ready(client: Pinecone) -> None:
    """Create a serverless index, wait for ready state, verify fields, then delete."""
    name = unique_name("idx")
    try:
        model = client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        assert model.name == name
        assert model.dimension == 2
        assert model.metric == "cosine"
        assert model.status.ready is True
        assert model.status.state == "Ready"
        assert model.spec.serverless is not None
        assert model.spec.serverless.cloud == "aws"
        assert model.spec.serverless.region == "us-east-1"
        assert model.deletion_protection == "disabled"
        assert isinstance(model.host, str)
        assert len(model.host) > 0
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# create-index — integrated (model-backed) dense
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(400)
@pytest.mark.xfail(
    reason="DX-0085: IndexModel missing embed field — API response embed is silently dropped by msgspec",
    strict=True,
)
def test_create_integrated_dense_index_becomes_ready(client: Pinecone) -> None:
    """Create an integrated dense index, wait for ready state, verify fields, then delete."""
    name = unique_name("int")
    try:
        model = client.indexes.create(
            name=name,
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="llama-text-embed-v2",
                    field_map={"text": "chunk_text"},
                    metric="cosine",
                ),
            ),
            timeout=300,
        )

        assert model.name == name
        assert model.status.ready is True
        assert model.status.state == "Ready"
        assert model.dimension == 1024
        assert model.metric == "cosine"
        assert model.vector_type == "dense"
        assert model.embed is not None
        assert model.embed.model == "llama-text-embed-v2"
        assert model.embed.field_map["text"] == "chunk_text"

        desc = client.indexes.describe(name)
        assert desc.name == name
        assert desc.status.ready is True
        assert desc.status.state == "Ready"
        assert desc.dimension == 1024
        assert desc.metric == "cosine"
        assert desc.vector_type == "dense"
        assert desc.embed is not None
        assert desc.embed.model == "llama-text-embed-v2"
        assert desc.embed.field_map["text"] == "chunk_text"
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# create-index — integrated (model-backed) sparse
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(400)
@pytest.mark.xfail(
    reason="DX-0085: IndexModel missing embed field — API response embed is silently dropped by msgspec",
    strict=True,
)
def test_create_integrated_sparse_index_becomes_ready(client: Pinecone) -> None:
    """Create an integrated sparse index, wait for ready state, verify fields, then delete."""
    name = unique_name("int")
    try:
        model = client.indexes.create(
            name=name,
            spec=IntegratedSpec(
                cloud="aws",
                region="us-east-1",
                embed=EmbedConfig(
                    model="pinecone-sparse-english-v0",
                    field_map={"text": "chunk_text"},
                    metric="dotproduct",
                ),
            ),
            timeout=300,
        )

        assert model.name == name
        assert model.status.ready is True
        assert model.status.state == "Ready"
        assert model.metric == "dotproduct"
        assert model.vector_type == "sparse"
        # Sparse indexes may not report a fixed dimension — just check the field is present
        # and its value is not an unexpected type if populated.
        assert model.dimension is None or isinstance(model.dimension, int)
        assert model.embed is not None
        assert model.embed.model == "pinecone-sparse-english-v0"
        assert model.embed.field_map["text"] == "chunk_text"

        desc = client.indexes.describe(name)
        assert desc.name == name
        assert desc.status.ready is True
        assert desc.status.state == "Ready"
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "sparse"
        assert desc.embed is not None
        assert desc.embed.model == "pinecone-sparse-english-v0"
        assert desc.embed.field_map["text"] == "chunk_text"
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# describe-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_describe_index_returns_full_model(client: Pinecone) -> None:
    """Create a serverless index, describe it, verify all IndexModel fields."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=4,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        desc = client.indexes.describe(name)

        assert isinstance(desc, IndexModel)
        assert desc.name == name
        assert desc.dimension == 4
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "dense"
        assert desc.deletion_protection == "disabled"

        # Status fields
        assert isinstance(desc.status, IndexStatus)
        assert desc.status.ready is True
        assert isinstance(desc.status.state, str)
        assert len(desc.status.state) > 0

        # Spec is serverless
        assert isinstance(desc.spec, IndexSpec)
        assert desc.spec.serverless is not None
        assert desc.spec.pod is None
        assert desc.spec.serverless.cloud == "aws"
        assert desc.spec.serverless.region == "us-east-1"

        # Host is a non-empty string
        assert isinstance(desc.host, str)
        assert len(desc.host) > 0
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-handle
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_index_handle_rest(client: Pinecone) -> None:
    """pc.index(name=...) returns a REST Index with the correct host."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # Get the expected host from describe
        desc = client.indexes.describe(name)
        expected_host = desc.host

        # Get an Index handle by name — triggers a describe call internally
        idx = client.index(name=name)

        assert isinstance(idx, Index)
        assert isinstance(idx.host, str)
        assert len(idx.host) > 0
        # Index normalizes host by prepending 'https://', so the raw describe
        # host (bare hostname) will be a suffix of idx.host
        assert expected_host in idx.host
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


@pytest.mark.integration
def test_index_handle_grpc(client: Pinecone) -> None:
    """pc.index(name=..., grpc=True) returns a GrpcIndex with the correct host."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # Get the expected host from describe
        desc = client.indexes.describe(name)
        expected_host = desc.host

        # Get a GrpcIndex handle by name
        idx = client.index(name=name, grpc=True)

        assert isinstance(idx, GrpcIndex)
        assert isinstance(idx.host, str)
        assert len(idx.host) > 0
        # GrpcIndex normalizes host similarly; bare hostname should appear in idx.host
        assert expected_host in idx.host
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-tags
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_index_with_tags(client: Pinecone) -> None:
    """Create a serverless index with tags and verify they are returned by describe."""
    name = unique_name("idx")
    tags = {"env": "integration-test", "version": "1"}
    try:
        model = client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            tags=tags,
            timeout=300,
        )

        # Tags should be present on the create response
        assert model.tags is not None
        assert model.tags.get("env") == "integration-test"
        assert model.tags.get("version") == "1"

        # Tags should also be present on describe
        desc = client.indexes.describe(name)
        assert desc.tags is not None
        assert desc.tags.get("env") == "integration-test"
        assert desc.tags.get("version") == "1"
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-exists
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_index_exists_returns_correct_bool(client: Pinecone) -> None:
    """indexes.exists() returns False before creation, True after, and False after deletion."""
    name = unique_name("idx")

    # Before creation: non-existent name → False
    assert client.indexes.exists(name) is False

    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # After creation: existing index → True
        assert client.indexes.exists(name) is True

        # Delete the index and wait for it to disappear
        client.indexes.delete(name, timeout=120)

        # After deletion: name no longer exists → False
        assert client.indexes.exists(name) is False
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# configure-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_configure_index_updates_tags(client: Pinecone) -> None:
    """configure() merges tags — add new tags, update existing tags, remove tags via empty string."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            tags={"env": "integration-test", "version": "1", "to-remove": "yes"},
            timeout=300,
        )

        # Add a new tag and update an existing tag
        client.indexes.configure(
            name,
            tags={"version": "2", "new-key": "new-val"},
        )

        desc = client.indexes.describe(name)
        assert desc.tags is not None
        assert desc.tags.get("env") == "integration-test"  # untouched
        assert desc.tags.get("version") == "2"  # updated
        assert desc.tags.get("new-key") == "new-val"  # added
        assert desc.tags.get("to-remove") == "yes"  # not yet removed

        # Remove a tag by setting its value to ""
        client.indexes.configure(
            name,
            tags={"to-remove": ""},
        )

        desc2 = client.indexes.describe(name)
        assert desc2.tags is not None
        assert "to-remove" not in desc2.tags or desc2.tags.get("to-remove") == ""
        assert desc2.tags.get("version") == "2"  # preserved from previous configure
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


@pytest.mark.integration
def test_configure_deletion_protection_toggle_rest(client: Pinecone) -> None:
    """configure() can enable and disable deletion protection; delete raises ForbiddenError when enabled."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # Enable deletion protection
        client.indexes.configure(name, deletion_protection="enabled")

        desc = client.indexes.describe(name)
        assert desc.deletion_protection == "enabled"

        # Attempting to delete a protected index must raise ForbiddenError (HTTP 403)
        with pytest.raises(ForbiddenError) as exc_info:
            client.indexes.delete(name)
        assert exc_info.value.status_code == 403

        # Disable deletion protection so the index can be cleaned up
        client.indexes.configure(name, deletion_protection="disabled")

        desc2 = client.indexes.describe(name)
        assert desc2.deletion_protection == "disabled"
    finally:
        # Ensure protection is off before deletion (in case test failed mid-way)
        try:
            client.indexes.configure(name, deletion_protection="disabled")
        except Exception:
            pass
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# delete with timeout=-1 (no-wait deletion) — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delete_index_timeout_minus1_returns_immediately(client: Pinecone) -> None:
    """indexes.delete(name, timeout=-1) returns None immediately without polling.

    Verifies claims:
    - unified-index-0057: deletion with timeout=-1 returns immediately without polling
    - unified-rs-0002: index deletion returns no response body (None)
    """
    from pinecone.errors import NotFoundError

    name = unique_name("idx")
    deleted = False
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # delete with timeout=-1 must return None immediately (no polling)
        result = client.indexes.delete(name, timeout=-1)
        deleted = True
        assert result is None  # unified-rs-0002: returns no response body

        # The index may still exist briefly (we didn't wait) — verify it eventually
        # disappears by polling the describe endpoint until NotFoundError
        import time

        start = time.monotonic()
        gone = False
        while time.monotonic() - start < 120:
            try:
                client.indexes.describe(name)
                time.sleep(5)
            except NotFoundError:
                gone = True
                break
        assert gone, f"Index '{name}' still exists 120s after delete(timeout=-1)"
    finally:
        if not deleted:
            cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# configure-index returns None and preserves unspecified fields
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_configure_returns_none_and_preserves_deletion_protection(client: Pinecone) -> None:
    """configure() always returns None; omitting deletion_protection leaves it unchanged.

    Verifies claims:
    - unified-index-0029: configure-index discards the API response and returns None
    - unified-index-0022: when configure is called without deletion_protection, the
      current value is preserved (the SDK omits the field from the PATCH body)
    """
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # Enable deletion protection — must return None
        result1 = client.indexes.configure(name, deletion_protection="enabled")
        assert result1 is None, "configure() must return None (unified-index-0029)"

        # Verify deletion protection is now "enabled"
        desc1 = client.indexes.describe(name)
        assert desc1.deletion_protection == "enabled"

        # Configure just a tag — no deletion_protection argument — must return None
        result2 = client.indexes.configure(name, tags={"test-key": "test-val"})
        assert result2 is None, "configure() must return None (unified-index-0029)"

        # Describe again: deletion_protection must still be "enabled" (preserved, not reset)
        desc2 = client.indexes.describe(name)
        assert desc2.deletion_protection == "enabled", (
            "deletion_protection must be preserved when configure() is called without it "
            "(unified-index-0022)"
        )
        # Also verify the tag was applied
        assert desc2.tags is not None
        assert desc2.tags.get("test-key") == "test-val"

    finally:
        # Ensure deletion protection is disabled before attempting to delete
        try:
            client.indexes.configure(name, deletion_protection="disabled")
        except Exception:
            pass
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# host-cache invalidation after delete
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delete_index_clears_host_cache_rest(client: Pinecone) -> None:
    """Deleting an index clears the cached host URL; pc.index(name) then raises NotFoundError.

    Verifies claims:
    - unified-index-0020: Deleting an index removes that index's cached host URL.

    Sequence:
    1. Create index (populates nothing in cache yet).
    2. Call pc.index(name) — triggers describe + caches the resolved host.
    3. Verify the cache entry now exists.
    4. Delete the index (default timeout — polls until fully gone, clears cache).
    5. Verify cache entry was removed.
    6. Call pc.index(name) again — cache miss → fresh describe → NotFoundError.
    """
    name = unique_name("idx")
    deleted = False
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # Step 2: resolve host via name — this populates the cache
        idx = client.index(name=name)
        assert isinstance(idx, Index)

        # Step 3: host should now be cached
        assert name in client._host_cache, (
            "Host must be cached after pc.index(name=name) (unified-index-0019)"
        )

        # Step 4: delete clears cache immediately then polls until gone
        client.indexes.delete(name)
        deleted = True

        # Step 5: cache entry must be gone
        assert name not in client._host_cache, (
            "Host cache must be cleared after delete() (unified-index-0020)"
        )

        # Step 6: cache miss → auto-describe → NotFoundError (index is gone)
        with pytest.raises(NotFoundError):
            client.index(name=name)

    finally:
        if not deleted:
            cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# schema parameter — flat vs nested format normalization — REST sync
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skip(
    reason=(
        "IT-0017: _normalize_schema() strips the 'fields' wrapper, sending flat format to the "
        "API. But the 2025-10 API version requires nested format {'fields': {...}}. Both flat and "
        "nested caller inputs are converted to flat by the SDK and then rejected with 422."
    )
)
def test_create_index_with_schema_normalization_rest(client: Pinecone) -> None:
    """create() accepts schema in both flat and nested formats, proving _normalize_schema() works.

    DISABLED: IT-0017 — _normalize_schema() incorrectly strips the 'fields' wrapper before
    sending to the API. The 2025-10 API requires nested {"fields": {...}} format inside
    spec.serverless.schema; the flat format is rejected with 422 Unprocessable Entity.

    Verifies claims:
    - unified-schema-0001: A schema specified as a flat map of field names to properties
      and a schema specified as a nested structure with a 'fields' wrapper key are parsed
      identically. The normalization is client-side: if nested format is accepted by the API,
      _normalize_schema() successfully unwrapped {"fields": {...}} before sending.
    - unified-schema-0002: Schemas can be included in index creation requests for serverless specs.

    Strategy: create two indexes — one with nested schema ({"fields": {...}}) and one with
    flat schema ({"field": {...}}). Both must be accepted by the API without error. The
    nested format requires normalization; the flat format is a passthrough.
    """
    name_nested = unique_name("idx")
    name_flat = unique_name("idx")
    try:
        # --- Step 1: nested schema format {"fields": {...}} ---
        # _normalize_schema() unwraps this to {"genre": {"filterable": True}} before the API call.
        result_nested = client.indexes.create(
            name=name_nested,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            schema={"fields": {"genre": {"filterable": True}}},
            timeout=300,
        )
        assert isinstance(result_nested, IndexModel), (
            "create() with nested schema must return an IndexModel"
        )
        assert result_nested.name == name_nested
        assert result_nested.status.ready is True

        # --- Step 2: flat schema format {"field": {...}} ---
        # _normalize_schema() passes this through unchanged.
        result_flat = client.indexes.create(
            name=name_flat,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            schema={"genre": {"filterable": True}},
            timeout=300,
        )
        assert isinstance(result_flat, IndexModel), (
            "create() with flat schema must return an IndexModel"
        )
        assert result_flat.name == name_flat
        assert result_flat.status.ready is True

    finally:
        cleanup_resource(lambda: client.indexes.delete(name_nested), name_nested, "index")
        cleanup_resource(lambda: client.indexes.delete(name_flat), name_flat, "index")


# ---------------------------------------------------------------------------
# IndexModel bracket access — unified-index-0026
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_index_model_bracket_access_on_real_describe(client: Pinecone) -> None:
    """IndexModel supports bracket access and containment check on a real describe() response.

    unified-index-0026: "The describe-index response supports both attribute and
    bracket access."

    All existing tests use only attribute access (model.name, model.metric …).
    This test verifies that the string-key bracket syntax (model['name']) and the
    'in' operator work correctly on a real API-deserialized IndexModel, and that
    accessing a non-existent key raises KeyError.

    Index creation uses timeout=-1 so the test does not wait for the index to be
    ready — describe() returns a valid IndexModel even in Initializing state.

    Area tag: index-model-bracket-access
    Transport: rest
    """
    index_name = unique_name("idx")
    try:
        # Create quickly — don't wait for ready; describe() works on any state
        client.indexes.create(
            name=index_name,
            dimension=2,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            timeout=-1,
        )

        model = client.indexes.describe(index_name)
        assert isinstance(model, IndexModel)

        # --- Bracket access equals attribute access ---
        assert model["name"] == model.name, "model['name'] must equal model.name"
        assert model["metric"] == model.metric, "model['metric'] must equal model.metric"
        assert model["host"] == model.host, "model['host'] must equal model.host"
        assert model["vector_type"] == model.vector_type, (
            "model['vector_type'] must equal model.vector_type"
        )
        assert model["deletion_protection"] == model.deletion_protection, (
            "model['deletion_protection'] must equal model.deletion_protection"
        )
        # dimension can be None for integrated indexes; assert equality regardless
        assert model["dimension"] == model.dimension, (
            "model['dimension'] must equal model.dimension"
        )

        # --- Specific field values ---
        assert model["name"] == index_name, "Bracket 'name' must match the created index name"
        assert model["metric"] == "cosine", "Bracket 'metric' must be 'cosine'"
        assert model["vector_type"] == "dense", "Bracket 'vector_type' must be 'dense'"
        assert model["deletion_protection"] == "disabled", (
            "Bracket 'deletion_protection' must be 'disabled'"
        )

        # --- Containment check ---
        for field in ("name", "metric", "host", "dimension", "deletion_protection", "vector_type"):
            assert field in model, f"'{field}' must be in IndexModel"
        assert "nonexistent_field_xyz" not in model, "Non-existent key must NOT be in IndexModel"

        # --- KeyError on missing key ---
        with pytest.raises(KeyError):
            _ = model["nonexistent_field_xyz"]

    finally:
        cleanup_resource(
            lambda: client.indexes.delete(index_name),
            index_name,
            "index",
        )
