"""Integration tests for index CRUD operations (async / REST async)."""

from __future__ import annotations

import pytest

from pinecone import AsyncIndex, AsyncPinecone
from pinecone.errors import ForbiddenError
from pinecone.models.indexes.index import IndexModel, IndexSpec, IndexStatus
from pinecone.models.indexes.specs import EmbedConfig, IntegratedSpec, ServerlessSpec
from tests.integration.conftest import async_cleanup_resource, unique_name

# ---------------------------------------------------------------------------
# list-indexes
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_list_indexes_returns_index_list(async_client: AsyncPinecone) -> None:
    """async pc.indexes.list() returns an IndexList that is iterable and supports len()."""
    result = await async_client.indexes.list()

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
@pytest.mark.anyio
async def test_create_serverless_index_becomes_ready(async_client: AsyncPinecone) -> None:
    """Create a serverless index asynchronously, wait for ready state, verify fields, then delete."""
    name = unique_name("idx")
    try:
        model = await async_client.indexes.create(
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
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(400)
async def test_create_integrated_dense_index_becomes_ready_async(
    async_client: AsyncPinecone,
) -> None:
    """Create an integrated dense index asynchronously, wait for ready state, verify fields, then delete."""
    name = unique_name("int")
    try:
        model = await async_client.indexes.create(
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

        described = await async_client.indexes.describe(name)
        assert described.name == model.name
        assert described.status.ready is True
        assert described.status.state == "Ready"
        assert described.dimension == 1024
        assert described.metric == "cosine"
        assert described.vector_type == "dense"
        assert described.embed is not None
        assert described.embed.model == "llama-text-embed-v2"
        assert described.embed.field_map["text"] == "chunk_text"
    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# describe-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_describe_index_returns_full_model(async_client: AsyncPinecone) -> None:
    """Create a serverless index asynchronously, describe it, verify all IndexModel fields."""
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            dimension=4,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        desc = await async_client.indexes.describe(name)

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
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-handle
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_index_handle_rest_async(async_client: AsyncPinecone) -> None:
    """async pc.index(name=...) returns an AsyncIndex with the correct host.

    AsyncPinecone.index() requires the host to be cached via a prior
    describe call. We call describe() first, which populates the host cache,
    then pc.index(name=name) should succeed.
    """
    name = unique_name("idx")
    idx = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # describe() caches the host in AsyncPinecone's host cache
        desc = await async_client.indexes.describe(name)
        expected_host = desc.host

        # Get an AsyncIndex handle by name (uses cached host)
        idx = await async_client.index(name=name)

        assert isinstance(idx, AsyncIndex)
        assert isinstance(idx.host, str)
        assert len(idx.host) > 0
        # AsyncIndex normalizes host by prepending 'https://', so the raw describe
        # host (bare hostname) will appear within idx.host
        assert expected_host in idx.host
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-tags
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_create_index_with_tags(async_client: AsyncPinecone) -> None:
    """Create a serverless index with tags asynchronously and verify they are returned by describe."""
    name = unique_name("idx")
    tags = {"env": "integration-test", "version": "1"}
    try:
        model = await async_client.indexes.create(
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
        desc = await async_client.indexes.describe(name)
        assert desc.tags is not None
        assert desc.tags.get("env") == "integration-test"
        assert desc.tags.get("version") == "1"
    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-exists
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_index_exists_returns_correct_bool(async_client: AsyncPinecone) -> None:
    """async indexes.exists() returns False before creation, True after, and False after deletion."""
    name = unique_name("idx")

    # Before creation: non-existent name → False
    assert await async_client.indexes.exists(name) is False

    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # After creation: existing index → True
        assert await async_client.indexes.exists(name) is True

        # Delete the index and wait for it to disappear
        await async_client.indexes.delete(name, timeout=120)

        # After deletion: name no longer exists → False
        assert await async_client.indexes.exists(name) is False
    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


@pytest.mark.integration
@pytest.mark.anyio
async def test_index_exists_with_empty_name_returns_false(async_client: AsyncPinecone) -> None:
    """Empty/whitespace names must short-circuit to False without a network call."""
    assert await async_client.indexes.exists("") is False
    assert await async_client.has_index("") is False


# ---------------------------------------------------------------------------
# configure-index
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_configure_index_updates_tags(async_client: AsyncPinecone) -> None:
    """async configure() merges tags — add new tags, update existing, remove via empty string."""
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            tags={"env": "integration-test", "version": "1", "to-remove": "yes"},
            timeout=300,
        )

        # Add a new tag and update an existing tag
        await async_client.indexes.configure(
            name,
            tags={"version": "2", "new-key": "new-val"},
        )

        desc = await async_client.indexes.describe(name)
        assert desc.tags is not None
        assert desc.tags.get("env") == "integration-test"  # untouched
        assert desc.tags.get("version") == "2"  # updated
        assert desc.tags.get("new-key") == "new-val"  # added
        assert desc.tags.get("to-remove") == "yes"  # not yet removed

        # Remove a tag by setting its value to ""
        await async_client.indexes.configure(
            name,
            tags={"to-remove": ""},
        )

        desc2 = await async_client.indexes.describe(name)
        assert desc2.tags is not None
        assert "to-remove" not in desc2.tags or desc2.tags.get("to-remove") == ""
        assert desc2.tags.get("version") == "2"  # preserved from previous configure
    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


@pytest.mark.integration
@pytest.mark.anyio
async def test_configure_deletion_protection_toggle_async(async_client: AsyncPinecone) -> None:
    """async configure() can enable/disable deletion protection; delete raises ForbiddenError when enabled."""
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # Enable deletion protection
        await async_client.indexes.configure(name, deletion_protection="enabled")

        desc = await async_client.indexes.describe(name)
        assert desc.deletion_protection == "enabled"

        # Attempting to delete a protected index must raise ForbiddenError (HTTP 403)
        with pytest.raises(ForbiddenError) as exc_info:
            await async_client.indexes.delete(name)
        assert exc_info.value.status_code == 403

        # Disable deletion protection so the index can be cleaned up
        await async_client.indexes.configure(name, deletion_protection="disabled")

        desc2 = await async_client.indexes.describe(name)
        assert desc2.deletion_protection == "disabled"
    finally:
        # Ensure protection is off before deletion (in case test failed mid-way)
        try:
            await async_client.indexes.configure(name, deletion_protection="disabled")
        except Exception:
            pass
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# delete with timeout=-1 (no-wait deletion) — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_delete_index_timeout_minus1_returns_immediately_async(
    async_client: AsyncPinecone,
) -> None:
    """async indexes.delete(name, timeout=-1) returns None immediately without polling.

    Verifies claims:
    - unified-index-0057: deletion with timeout=-1 returns immediately without polling
    - unified-rs-0002: index deletion returns no response body (None)
    """
    import asyncio

    from pinecone.errors import NotFoundError

    name = unique_name("idx")
    deleted = False
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # delete with timeout=-1 must return None immediately (no polling)
        result = await async_client.indexes.delete(name, timeout=-1)
        deleted = True
        assert result is None  # unified-rs-0002: returns no response body

        # Poll until the index is gone (verify the deletion was actually triggered)
        gone = False
        for _ in range(24):  # up to 120 seconds (24 * 5s)
            try:
                await async_client.indexes.describe(name)
                await asyncio.sleep(5)
            except NotFoundError:
                gone = True
                break
        assert gone, f"Index '{name}' still exists 120s after async delete(timeout=-1)"
    finally:
        if not deleted:
            await async_cleanup_resource(
                lambda: async_client.indexes.delete(name),
                name,
                "index",
            )


# ---------------------------------------------------------------------------
# configure-index returns None and preserves unspecified fields
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_configure_returns_none_and_preserves_deletion_protection_async(
    async_client: AsyncPinecone,
) -> None:
    """async configure() always returns None; omitting deletion_protection leaves it unchanged.

    Verifies claims:
    - unified-index-0029: configure-index discards the API response and returns None
    - unified-index-0022: when configure is called without deletion_protection, the
      current value is preserved (the SDK omits the field from the PATCH body)
    """
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # Enable deletion protection — must return None
        result1 = await async_client.indexes.configure(name, deletion_protection="enabled")
        assert result1 is None, "configure() must return None (unified-index-0029)"

        # Verify deletion protection is now "enabled"
        desc1 = await async_client.indexes.describe(name)
        assert desc1.deletion_protection == "enabled"

        # Configure just a tag — no deletion_protection argument — must return None
        result2 = await async_client.indexes.configure(name, tags={"test-key": "test-val"})
        assert result2 is None, "configure() must return None (unified-index-0029)"

        # Describe again: deletion_protection must still be "enabled" (preserved, not reset)
        desc2 = await async_client.indexes.describe(name)
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
            await async_client.indexes.configure(name, deletion_protection="disabled")
        except Exception:
            pass
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# async index factory requires prior describe; delete clears cache
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_async_index_factory_auto_resolves_on_cache_miss_rest_async(
    async_client: AsyncPinecone,
) -> None:
    """AsyncPinecone.index(name) auto-resolves the host via describe() on cache miss.

    Verifies claims:
    - unified-index-0020: Deleting an index removes that index's cached host URL.
    - unified-index-0024: Both sync and async index clients auto-resolve via
      describe() on cache miss; there is no asymmetry between the two.

    Sequence:
    1. Create index.
    2. Pop the host cache entry to simulate a cold-cache scenario.
    3. Call await async_client.index(name) — cache miss → describe is called →
       AsyncIndex returned; cache is repopulated.
    4. Delete the index (clears cache, polls until gone).
    5. Call await async_client.index(name) after deletion — describe returns 404
       → NotFoundError raised.
    """
    from pinecone.errors import NotFoundError

    name = unique_name("idx")
    deleted = False
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        # create() with timeout polling already populated the cache via describe.
        # Clear it to simulate a cold-cache scenario for the factory test.
        async_client._host_cache.pop(name, None)
        assert name not in async_client._host_cache

        # Step 3: cache miss → auto-resolve via describe → AsyncIndex returned
        idx = await async_client.index(name=name)
        assert isinstance(idx, AsyncIndex)
        assert name in async_client._host_cache, (
            "Host must be cached after auto-resolve on cache miss (unified-index-0024)"
        )

        # Step 4: delete clears cache immediately and polls until gone
        await async_client.indexes.delete(name)
        deleted = True

        assert name not in async_client._host_cache, (
            "Host cache must be cleared after delete() (unified-index-0020)"
        )

        # Step 5: cache miss after delete → describe returns 404 → NotFoundError
        with pytest.raises(NotFoundError):
            await async_client.index(name=name)

    finally:
        if not deleted:
            await async_cleanup_resource(
                lambda: async_client.indexes.delete(name),
                name,
                "index",
            )


# ---------------------------------------------------------------------------
# schema parameter — flat vs nested format normalization — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.skip(
    reason=(
        "IT-0017: same root cause as sync variant — _normalize_schema() strips 'fields' wrapper, "
        "sending flat format which the 2025-10 API rejects with 422."
    )
)
async def test_create_index_with_schema_normalization_async(
    async_client: AsyncPinecone,
) -> None:
    """Async create() accepts schema in both flat and nested formats.

    DISABLED: IT-0017 — same root cause as sync variant.

    Async variant of test_create_index_with_schema_normalization_rest. Verifies:
    - unified-schema-0001: nested schema {"fields": {...}} is normalized to flat format
      identically to flat schema {"field": {...}} before the API call.
    - unified-schema-0002: schemas can be included in serverless index creation requests.
    """
    name_nested = unique_name("idx")
    name_flat = unique_name("idx")
    try:
        # --- Step 1: nested schema format ---
        result_nested = await async_client.indexes.create(
            name=name_nested,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            schema={"fields": {"genre": {"filterable": True}}},
            timeout=300,
        )
        assert isinstance(result_nested, IndexModel), (
            "async create() with nested schema must return an IndexModel"
        )
        assert result_nested.name == name_nested
        assert result_nested.status.ready is True

        # --- Step 2: flat schema format ---
        result_flat = await async_client.indexes.create(
            name=name_flat,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            schema={"genre": {"filterable": True}},
            timeout=300,
        )
        assert isinstance(result_flat, IndexModel), (
            "async create() with flat schema must return an IndexModel"
        )
        assert result_flat.name == name_flat
        assert result_flat.status.ready is True

    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name_nested), name_nested, "index"
        )
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name_flat), name_flat, "index"
        )


# ---------------------------------------------------------------------------
# IndexModel bracket access — unified-index-0026
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.timeout(300)
async def test_index_model_bracket_access_on_real_describe_async(
    async_client: AsyncPinecone,
) -> None:
    """Async variant: IndexModel supports bracket access on a real describe() response.

    Verifies unified-index-0026 (bracket access) on the async transport path.

    Area tag: index-model-bracket-access
    Transport: rest-async
    """
    index_name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=index_name,
            dimension=2,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            timeout=-1,
        )

        model = await async_client.indexes.describe(index_name)
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
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(index_name),
            index_name,
            "index",
        )
