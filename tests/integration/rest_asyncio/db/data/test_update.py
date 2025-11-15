import pytest
from pinecone import Vector
from .conftest import build_asyncioindex_client, poll_until_lsn_reconciled_async
from tests.integration.helpers import random_string, embedding_values


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

    async def test_update_with_filter_and_dry_run(self, index_host, dimension, target_namespace):
        """Test update with filter and dry_run=True to verify matched_records and updated_records are returned."""
        asyncio_idx = build_asyncioindex_client(index_host)

        # Upsert vectors with different genres
        upsert1 = await asyncio_idx.upsert(
            vectors=[
                Vector(
                    id=str(i),
                    values=embedding_values(dimension),
                    metadata={"genre": "comedy" if i % 2 == 0 else "drama", "status": "active"},
                )
                for i in range(10)
            ],
            namespace=target_namespace,
            batch_size=10,
            show_progress=False,
        )

        await poll_until_lsn_reconciled_async(
            asyncio_idx, upsert1._response_info, namespace=target_namespace
        )

        # Test dry_run=True - should return matched_records without updating
        dry_run_response = await asyncio_idx.update(
            filter={"genre": {"$eq": "comedy"}},
            set_metadata={"status": "updated"},
            dry_run=True,
            namespace=target_namespace,
        )

        # Verify matched_records is returned and correct (5 comedy vectors)
        assert dry_run_response.matched_records is not None
        assert dry_run_response.matched_records == 5
        # In dry run, updated_records should be 0 or None since no records are actually updated
        assert dry_run_response.updated_records is None or dry_run_response.updated_records == 0

        # Verify the vectors were NOT actually updated (dry run)
        fetched_before = await asyncio_idx.fetch(
            ids=["0", "2", "4", "6", "8"], namespace=target_namespace
        )
        for vec_id in ["0", "2", "4", "6", "8"]:
            assert fetched_before.vectors[vec_id].metadata.get("status") == "active"

        # Now do the actual update
        update_response = await asyncio_idx.update(
            filter={"genre": {"$eq": "comedy"}},
            set_metadata={"status": "updated"},
            namespace=target_namespace,
        )

        # Verify matched_records and updated_records are returned
        assert update_response.matched_records is not None
        assert update_response.matched_records == 5
        # updated_records should match the number of records actually updated (if returned by API)
        if update_response.updated_records is not None:
            assert update_response.updated_records == 5

        await poll_until_lsn_reconciled_async(
            asyncio_idx, update_response._response_info, namespace=target_namespace
        )

        # Verify the vectors were actually updated
        fetched_after = await asyncio_idx.fetch(
            ids=["0", "2", "4", "6", "8"], namespace=target_namespace
        )
        for vec_id in ["0", "2", "4", "6", "8"]:
            assert fetched_after.vectors[vec_id].metadata.get("status") == "updated"
            assert fetched_after.vectors[vec_id].metadata.get("genre") == "comedy"

        await asyncio_idx.close()
