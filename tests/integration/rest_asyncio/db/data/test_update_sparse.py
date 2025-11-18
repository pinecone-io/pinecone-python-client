import pytest
from pinecone import Vector, SparseValues
from .conftest import build_asyncioindex_client, poll_until_lsn_reconciled_async
from tests.integration.helpers import random_string, embedding_values


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
class TestAsyncioUpdateSparse:
    async def test_update_values(self, sparse_index_host, target_namespace):
        asyncio_idx = build_asyncioindex_client(sparse_index_host)

        upsert1 = await asyncio_idx.upsert(
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

        await poll_until_lsn_reconciled_async(
            asyncio_idx, upsert1._response_info, namespace=target_namespace
        )

        # Update values
        new_sparse_values = {"indices": [j for j in range(100)], "values": embedding_values(100)}
        update1 = await asyncio_idx.update(
            id="1", sparse_values=new_sparse_values, namespace=target_namespace
        )

        await poll_until_lsn_reconciled_async(
            asyncio_idx, update1._response_info, namespace=target_namespace
        )

        fetch_updated = await asyncio_idx.fetch(ids=["1"], namespace=target_namespace)
        assert fetch_updated.vectors["1"].sparse_values.values[0] == pytest.approx(
            new_sparse_values["values"][0], 0.01
        )
        assert len(fetch_updated.vectors["1"].sparse_values.values) == 100

        fetched_vec = await asyncio_idx.fetch(ids=["1"], namespace=target_namespace)
        assert len(fetched_vec.vectors["1"].sparse_values.values) == 100
        await asyncio_idx.close()

    async def test_update_metadata(self, sparse_index_host, dimension, target_namespace):
        asyncio_idx = build_asyncioindex_client(sparse_index_host)

        sparse_values = SparseValues(indices=[j for j in range(100)], values=embedding_values(100))
        upsert1 = await asyncio_idx.upsert(
            vectors=[
                Vector(id=str(i), sparse_values=sparse_values, metadata={"genre": "action"})
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
            id="2", set_metadata={"genre": "comedy"}, namespace=target_namespace
        )

        await poll_until_lsn_reconciled_async(
            asyncio_idx, update1._response_info, namespace=target_namespace
        )

        fetch_updated = await asyncio_idx.fetch(ids=["2"], namespace=target_namespace)
        assert fetch_updated.vectors["2"].metadata == {"genre": "comedy"}
        await asyncio_idx.close()
