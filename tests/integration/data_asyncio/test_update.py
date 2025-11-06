import pytest
from pinecone import Vector
from .conftest import build_asyncioindex_client, poll_until_lsn_reconciled_async
from ..helpers import random_string, embedding_values


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
class TestAsyncioUpdate:
    async def test_update_values(self, index_host, dimension, target_namespace):
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
            asyncio_idx, upsert1._response_info, namespace=target_namespace
        )

        # Update values
        new_values = embedding_values(dimension)
        update1 = await asyncio_idx.update(id="1", values=new_values, namespace=target_namespace)

        await poll_until_lsn_reconciled_async(
            asyncio_idx, update1._response_info, namespace=target_namespace
        )

        fetched_vec = await asyncio_idx.fetch(ids=["1"], namespace=target_namespace)
        assert fetched_vec.vectors["1"].values[0] == pytest.approx(new_values[0], 0.01)
        assert fetched_vec.vectors["1"].values[1] == pytest.approx(new_values[1], 0.01)
        await asyncio_idx.close()

    async def test_update_metadata(self, index_host, dimension, target_namespace):
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
            asyncio_idx, upsert1._response_info, namespace=target_namespace
        )

        # Update metadata
        update1 = await asyncio_idx.update(
            id="2",
            values=embedding_values(dimension),
            set_metadata={"genre": "comedy"},
            namespace=target_namespace,
        )

        await poll_until_lsn_reconciled_async(
            asyncio_idx, update1._response_info, namespace=target_namespace
        )

        fetched_vec = await asyncio_idx.fetch(ids=["2"], namespace=target_namespace)
        assert fetched_vec.vectors["2"].metadata == {"genre": "comedy"}
        await asyncio_idx.close()
