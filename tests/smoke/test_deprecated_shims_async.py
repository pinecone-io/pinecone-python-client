"""Priority-2 smoke test — deprecated flat-method shims (async).

Mirror of ``test_deprecated_shims_sync.py`` for ``AsyncPinecone``.

Punchlist coverage (async, deprecated):

- create_index, describe_index, list_indexes, has_index, configure_index, delete_index
- create_index_for_model
- list_restore_jobs, describe_restore_job
- IndexAsyncio() factory (constructor only)
- create_collection / list_collections / describe_collection / delete_collection
  (gated behind ``SMOKE_RUN_POD_SHIMS=1``)

There is no ``Index()`` factory on AsyncPinecone — only ``IndexAsyncio()``.
Backup shims are out of scope.
"""

from __future__ import annotations

import os

import pytest

from pinecone import (
    AsyncIndex,
    AsyncPinecone,
    IndexEmbed,
    NotFoundError,
    PineconeApiException,
    PodSpec,
    ServerlessSpec,
)
from tests.smoke.conftest import (
    SMOKE_PREFIX,
    async_cleanup_resource,
    async_ensure_index_deleted,
    unique_name,
)
from tests.smoke.helpers import capture_deprecation_warning

CLOUD = "aws"
REGION = "us-east-1"
EMBED_MODEL = "multilingual-e5-large"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_deprecated_shims_smoke_async(api_key: str) -> None:
    """Sweep every AsyncPinecone flat-method shim."""
    async_client = AsyncPinecone(api_key=api_key)
    serverless_name = unique_name(f"{SMOKE_PREFIX}-shim-srv-async")
    integrated_name = unique_name(f"{SMOKE_PREFIX}-shim-int-async")
    pod_name = ""
    collection_name = ""

    try:
        # ----- create_index (serverless) -----
        with capture_deprecation_warning("create_index"):
            await async_client.create_index(
                name=serverless_name,
                spec=ServerlessSpec(cloud=CLOUD, region=REGION),
                dimension=8,
                metric="cosine",
            )

        with capture_deprecation_warning("describe_index"):
            described = await async_client.describe_index(serverless_name)
        assert described.name == serverless_name

        with capture_deprecation_warning("list_indexes"):
            listing = await async_client.list_indexes()
        assert any(i.name == serverless_name for i in listing.indexes)

        with capture_deprecation_warning("has_index"):
            exists = await async_client.has_index(serverless_name)
        assert exists is True

        with capture_deprecation_warning("configure_index"):
            await async_client.configure_index(
                serverless_name, tags={"env": "smoke-async"}
            )

        # ----- IndexAsyncio() factory (constructor only) -----
        with capture_deprecation_warning("IndexAsyncio"):
            async_idx = async_client.IndexAsyncio(host=described.host)
        assert isinstance(async_idx, AsyncIndex)

        with capture_deprecation_warning("delete_index"):
            await async_client.delete_index(serverless_name)
        serverless_name = ""

        # ----- create_index_for_model (integrated) -----
        with capture_deprecation_warning("create_index_for_model"):
            await async_client.create_index_for_model(
                name=integrated_name,
                cloud=CLOUD,
                region=REGION,
                embed=IndexEmbed(
                    model=EMBED_MODEL,
                    field_map={"text": "chunk_text"},
                ),
            )
        with capture_deprecation_warning("delete_index"):
            await async_client.delete_index(integrated_name)
        integrated_name = ""

        # ----- list_restore_jobs / describe_restore_job -----
        with capture_deprecation_warning("list_restore_jobs"):
            jobs = await async_client.list_restore_jobs(limit=10)
        if jobs.data:
            with capture_deprecation_warning("describe_restore_job"):
                await async_client.describe_restore_job(
                    job_id=jobs.data[0].restore_job_id
                )
        else:
            with capture_deprecation_warning("describe_restore_job"):
                with pytest.raises((NotFoundError, PineconeApiException)):
                    await async_client.describe_restore_job(
                        job_id=f"{SMOKE_PREFIX}-no-such-job-async"
                    )

        # ----- pod + collections shims (gated) -----
        if os.getenv("SMOKE_RUN_POD_SHIMS") == "1":
            pod_name = unique_name(f"{SMOKE_PREFIX}-shim-pod-async")
            collection_name = unique_name(f"{SMOKE_PREFIX}-shim-col-async")
            with capture_deprecation_warning("create_index"):
                await async_client.create_index(
                    name=pod_name,
                    spec=PodSpec(environment="us-east-1-aws", pod_type="p1.x1", pods=1),
                    dimension=8,
                    metric="cosine",
                )
            with capture_deprecation_warning("create_collection"):
                await async_client.create_collection(name=collection_name, source=pod_name)
            with capture_deprecation_warning("list_collections"):
                cols = await async_client.list_collections()
            assert collection_name in cols.names()
            with capture_deprecation_warning("describe_collection"):
                await async_client.describe_collection(collection_name)
            with capture_deprecation_warning("delete_collection"):
                await async_client.delete_collection(collection_name)
            collection_name = ""
            with capture_deprecation_warning("delete_index"):
                await async_client.delete_index(pod_name)
            pod_name = ""
    finally:
        if serverless_name:
            await async_ensure_index_deleted(async_client, serverless_name)
        if integrated_name:
            await async_ensure_index_deleted(async_client, integrated_name)
        if collection_name:
            await async_cleanup_resource(
                lambda: async_client.collections.delete(collection_name),
                collection_name,
                "collection",
            )
        if pod_name:
            await async_ensure_index_deleted(async_client, pod_name)
        await async_client.close()
