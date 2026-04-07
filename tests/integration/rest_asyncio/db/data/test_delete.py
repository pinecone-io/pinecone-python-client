import pytest
from .conftest import build_asyncioindex_client, poll_until_lsn_reconciled_async
from tests.integration.helpers import random_string, embedding_values


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
async def test_delete_by_ids(index_host, dimension, target_namespace):
    asyncio_idx = build_asyncioindex_client(index_host)
    try:
        ids = [f"del-{i}" for i in range(3)]
        vectors = [(id, embedding_values(dimension)) for id in ids]
        upsert_resp = await asyncio_idx.upsert(vectors=vectors, namespace=target_namespace)
        await poll_until_lsn_reconciled_async(
            asyncio_idx, upsert_resp._response_info, namespace=target_namespace
        )

        # Delete a subset of vectors by id
        delete_resp = await asyncio_idx.delete(ids=ids[:2], namespace=target_namespace)
        assert delete_resp is None or isinstance(delete_resp, dict)

        # Remaining vector should still be fetchable
        fetched = await asyncio_idx.fetch(ids=ids, namespace=target_namespace)
        assert ids[2] in fetched.vectors
        assert ids[0] not in fetched.vectors
        assert ids[1] not in fetched.vectors
    finally:
        await asyncio_idx.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
async def test_delete_all(index_host, dimension, target_namespace):
    asyncio_idx = build_asyncioindex_client(index_host)
    try:
        vectors = [(f"delall-{i}", embedding_values(dimension)) for i in range(3)]
        upsert_resp = await asyncio_idx.upsert(vectors=vectors, namespace=target_namespace)
        await poll_until_lsn_reconciled_async(
            asyncio_idx, upsert_resp._response_info, namespace=target_namespace
        )

        # Delete all vectors in namespace — this is the exact call from issue #564
        delete_resp = await asyncio_idx.delete(namespace=target_namespace, delete_all=True)
        assert delete_resp is None or isinstance(delete_resp, dict)
    finally:
        await asyncio_idx.close()
