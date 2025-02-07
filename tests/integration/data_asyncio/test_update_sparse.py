import pytest
from pinecone import Vector
from .conftest import build_asyncioindex_client, poll_for_freshness, wait_until
from ..helpers import random_string, embedding_values


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
class TestAsyncioUpdateSparse:
    async def test_update_values(self, sparse_index_host, target_namespace):
        asyncio_idx = build_asyncioindex_client(sparse_index_host)

        await asyncio_idx.upsert(
            vectors=[
                Vector(
                    id=str(i),
                    sparse_values={
                        "indices": [j for j in range(100)],
                        "values": embedding_values(100),
                    },
                    metadata={"genre": "action"},
                )
                for i in range(100)
            ],
            namespace=target_namespace,
            batch_size=10,
            show_progress=False,
        )

        await poll_for_freshness(asyncio_idx, target_namespace, 100)

        # Update values
        new_sparse_values = {"indices": [j for j in range(100)], "values": embedding_values(100)}
        await asyncio_idx.update(
            id="1", sparse_values=new_sparse_values, namespace=target_namespace
        )

        # Wait until the update is reflected in the first value of the vector
        async def wait_condition():
            fetched_vec = await asyncio_idx.fetch(ids=["1"], namespace=target_namespace)
            return fetched_vec.vectors["1"].sparse_values.values[0] == pytest.approx(
                new_sparse_values["values"][0], 0.01
            )

        await wait_until(wait_condition, timeout=180, interval=5)

        fetched_vec = await asyncio_idx.fetch(ids=["1"], namespace=target_namespace)
        assert len(fetched_vec.vectors["1"].sparse_values.values) == 100
        await asyncio_idx.close()

        # #  Check that all the values are updated
        # for i in range(100):
        #     assert fetched_vec.vectors["1"].sparse_values.values[i] == pytest.approx(
        #         new_sparse_values["values"][i], 0.01
        #     )

    @pytest.mark.skip(reason="Needs troubleshooting, possible bug")
    async def test_update_metadata(self, sparse_index_host, dimension, target_namespace):
        asyncio_idx = build_asyncioindex_client(sparse_index_host)

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

        await wait_until(wait_condition, timeout=60, interval=5)

        fetched_vec = await asyncio_idx.fetch(ids=["2"], namespace=target_namespace)
        assert fetched_vec.vectors["2"].metadata == {"genre": "comedy"}
        await asyncio_idx.close()
