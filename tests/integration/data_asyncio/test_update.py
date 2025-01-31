import pytest
from pinecone import Vector
from .conftest import build_asyncioindex_client, poll_for_freshness, wait_until
from ..helpers import random_string, embedding_values


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
class TestAsyncioUpdate:
    async def test_update_values(self, index_host, dimension, target_namespace):
        asyncio_idx = build_asyncioindex_client(index_host)

        await asyncio_idx.upsert(
            vectors=[
                Vector(id=str(i), values=embedding_values(dimension), metadata={"genre": "action"})
                for i in range(100)
            ],
            namespace=target_namespace,
            batch_size=10,
            show_progress=False,
        )

        await poll_for_freshness(asyncio_idx, target_namespace, 100)

        # Update values
        new_values = embedding_values(dimension)
        await asyncio_idx.update(id="1", values=new_values, namespace=target_namespace)

        async def wait_condition():
            fetched_vec = await asyncio_idx.fetch(ids=["1"], namespace=target_namespace)
            return fetched_vec.vectors["1"].values[0] == pytest.approx(new_values[0], 0.01)

        await wait_until(wait_condition, timeout=180, interval=10)

        fetched_vec = await asyncio_idx.fetch(ids=["1"], namespace=target_namespace)
        assert fetched_vec.vectors["1"].values[0] == pytest.approx(new_values[0], 0.01)
        assert fetched_vec.vectors["1"].values[1] == pytest.approx(new_values[1], 0.01)

    @pytest.mark.skip(reason="Needs troubleshooting, possible bug")
    async def test_update_metadata(self, index_host, dimension, target_namespace):
        asyncio_idx = build_asyncioindex_client(index_host)

        await asyncio_idx.upsert(
            vectors=[
                Vector(id=str(i), values=embedding_values(dimension), metadata={"genre": "action"})
                for i in range(100)
            ],
            namespace=target_namespace,
            batch_size=10,
            show_progress=False,
        )

        await poll_for_freshness(asyncio_idx, target_namespace, 100)

        # Update metadata
        await asyncio_idx.update(
            id="2", values=embedding_values(dimension), set_metadata={"genre": "comedy"}
        )

        async def wait_condition():
            fetched_vec = await asyncio_idx.fetch(ids=["2"], namespace=target_namespace)
            return fetched_vec.vectors["2"].metadata == {"genre": "comedy"}

        await wait_until(wait_condition, timeout=60, interval=10)

        fetched_vec = await asyncio_idx.fetch(ids=["1", "2"], namespace=target_namespace)
        assert fetched_vec.vectors["2"].metadata == {"genre": "comedy"}
