import pytest
from pinecone import Vector
from .conftest import build_asyncioindex_client, poll_until_lsn_reconciled_async
from ..helpers import random_string, embedding_values


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
async def test_list(index_host, dimension, target_namespace):
    asyncio_idx = build_asyncioindex_client(index_host)

    upsert1 = await asyncio_idx.upsert(
        vectors=[
            Vector(id=str(i), values=embedding_values(dimension), metadata={"genre": "action"})
            for i in range(100)
        ],
        namespace=target_namespace,
        batch_size=10,
        show_progress=False,
    )

    await poll_until_lsn_reconciled_async(
        asyncio_idx,
        target_lsn=upsert1._response_info.get("lsn_committed"),
        namespace=target_namespace,
    )

    # List all vectors
    async for ids_list in asyncio_idx.list(namespace=target_namespace, limit=11, prefix="9"):
        assert set(ids_list) == set(
            ("9", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99")
        )

    pages = 0
    async for ids_list in asyncio_idx.list(namespace=target_namespace, limit=4):
        pages += 1
        assert len(ids_list) <= 4
    assert pages == 25
    await asyncio_idx.close()
