"""Priority-2 smoke test — deprecated flat-method shims (sync).

Bumped to priority 2 (right after inference) because launch-day usage will
be dominated by callers migrating from v1, who lean on the shim surface.
If shims break, the upgrade path breaks for most users.

Each shim is exercised inside ``capture_deprecation_warning`` so we get a
hard failure when a method silently stops emitting ``DeprecationWarning``.

Punchlist coverage (sync, deprecated):

- pc.create_index, describe_index, list_indexes, has_index, configure_index, delete_index
- pc.create_index_for_model
- pc.list_restore_jobs, describe_restore_job
- pc.Index() factory
- pc.IndexAsyncio() factory (constructor only)
- pc.create_collection / list_collections / describe_collection / delete_collection
  (gated behind ``SMOKE_RUN_POD_SHIMS=1`` because the pod index is slow)

Backup shims are intentionally omitted — they require an external data
integration setup that is out of scope for smoke tests.
"""

from __future__ import annotations

import os

import pytest

from pinecone import (
    AsyncIndex,
    IndexEmbed,
    NotFoundError,
    Pinecone,
    PineconeApiException,
    PodSpec,
    ServerlessSpec,
)
from pinecone.index import Index
from tests.smoke.conftest import (
    SMOKE_PREFIX,
    cleanup_resource,
    ensure_index_deleted,
    unique_name,
)
from tests.smoke.helpers import capture_deprecation_warning

CLOUD = "aws"
REGION = "us-east-1"
EMBED_MODEL = "multilingual-e5-large"


@pytest.mark.smoke
def test_deprecated_shims_smoke(client: Pinecone) -> None:
    """Sweep every Pinecone flat-method shim and assert each emits a warning."""
    serverless_name = unique_name(f"{SMOKE_PREFIX}-shim-srv")
    integrated_name = unique_name(f"{SMOKE_PREFIX}-shim-int")
    # Pod / collection names are only generated inside the gated block.
    pod_name = ""
    collection_name = ""

    try:
        # ----- create_index (serverless) -----
        with capture_deprecation_warning("create_index"):
            client.create_index(
                name=serverless_name,
                spec=ServerlessSpec(cloud=CLOUD, region=REGION),
                dimension=8,
                metric="cosine",
            )

        # ----- describe_index -----
        with capture_deprecation_warning("describe_index"):
            described = client.describe_index(serverless_name)
        assert described.name == serverless_name

        # ----- list_indexes -----
        with capture_deprecation_warning("list_indexes"):
            listing = client.list_indexes()
        assert any(i.name == serverless_name for i in listing.indexes)

        # ----- has_index -----
        with capture_deprecation_warning("has_index"):
            exists = client.has_index(serverless_name)
        assert exists is True

        # ----- configure_index -----
        with capture_deprecation_warning("configure_index"):
            client.configure_index(serverless_name, tags={"env": "smoke"})

        # ----- pc.Index() factory -----
        with capture_deprecation_warning("Pinecone.Index"):
            idx_via_factory = client.Index(name=serverless_name)
        assert isinstance(idx_via_factory, Index)
        idx_via_factory.close()

        # ----- pc.IndexAsyncio() factory (constructor only — don't await) -----
        with capture_deprecation_warning("IndexAsyncio"):
            async_idx = client.IndexAsyncio(host=described.host)
        assert isinstance(async_idx, AsyncIndex)

        # ----- delete_index -----
        with capture_deprecation_warning("delete_index"):
            client.delete_index(serverless_name)
        # After delete, mark as already cleaned so finally doesn't double-call.
        serverless_name = ""

        # ----- create_index_for_model (integrated) -----
        with capture_deprecation_warning("create_index_for_model"):
            client.create_index_for_model(
                name=integrated_name,
                cloud=CLOUD,
                region=REGION,
                embed=IndexEmbed(
                    model=EMBED_MODEL,
                    field_map={"text": "chunk_text"},
                ),
            )
        with capture_deprecation_warning("delete_index"):
            client.delete_index(integrated_name)
        integrated_name = ""

        # ----- list_restore_jobs / describe_restore_job -----
        with capture_deprecation_warning("list_restore_jobs"):
            jobs = client.list_restore_jobs(limit=10)
        if jobs.data:
            with capture_deprecation_warning("describe_restore_job"):
                client.describe_restore_job(job_id=jobs.data[0].restore_job_id)
        else:
            # Confirm the call still exercises the deprecation path even when
            # the lookup fails — describe with a known-bogus id.
            with capture_deprecation_warning("describe_restore_job"):
                with pytest.raises((NotFoundError, PineconeApiException)):
                    client.describe_restore_job(job_id=f"{SMOKE_PREFIX}-no-such-job")

        # ----- pod + collections shims (gated; very slow) -----
        if os.getenv("SMOKE_RUN_POD_SHIMS") == "1":
            pod_name = unique_name(f"{SMOKE_PREFIX}-shim-pod")
            collection_name = unique_name(f"{SMOKE_PREFIX}-shim-col")
            with capture_deprecation_warning("create_index"):
                client.create_index(
                    name=pod_name,
                    spec=PodSpec(environment="us-east-1-aws", pod_type="p1.x1", pods=1),
                    dimension=8,
                    metric="cosine",
                )
            with capture_deprecation_warning("create_collection"):
                client.create_collection(name=collection_name, source=pod_name)
            with capture_deprecation_warning("list_collections"):
                cols = client.list_collections()
            assert collection_name in cols.names()
            with capture_deprecation_warning("describe_collection"):
                client.describe_collection(collection_name)
            with capture_deprecation_warning("delete_collection"):
                client.delete_collection(collection_name)
            collection_name = ""
            with capture_deprecation_warning("delete_index"):
                client.delete_index(pod_name)
            pod_name = ""
    finally:
        # Best-effort cleanup of anything still alive.
        if serverless_name:
            ensure_index_deleted(client, serverless_name)
        if integrated_name:
            ensure_index_deleted(client, integrated_name)
        if collection_name:
            cleanup_resource(
                lambda: client.collections.delete(collection_name),
                collection_name,
                "collection",
            )
        if pod_name:
            ensure_index_deleted(client, pod_name)
        client.close()
