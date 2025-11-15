import pytest
from pinecone import Vector
from tests.integration.helpers import poll_until_lsn_reconciled, embedding_values, random_string


@pytest.fixture(scope="session")
def update_namespace():
    return random_string(10)


class TestUpdate:
    def test_update_with_filter_and_dry_run(self, idx, update_namespace):
        """Test update with filter and dry_run=True to verify matched_records and updated_records are returned."""
        target_namespace = update_namespace

        # Upsert vectors with different genres
        upsert1 = idx.upsert(
            vectors=[
                Vector(
                    id=str(i),
                    values=embedding_values(),
                    metadata={"genre": "comedy" if i % 2 == 0 else "drama", "status": "active"},
                )
                for i in range(10)
            ],
            namespace=target_namespace,
        )

        poll_until_lsn_reconciled(idx, upsert1._response_info, namespace=target_namespace)

        # Test dry_run=True - should return matched_records without updating
        dry_run_response = idx.update(
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
        fetched_before = idx.fetch(ids=["0", "2", "4", "6", "8"], namespace=target_namespace)
        for vec_id in ["0", "2", "4", "6", "8"]:
            assert fetched_before.vectors[vec_id].metadata.get("status") == "active"

        # Now do the actual update
        update_response = idx.update(
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

        poll_until_lsn_reconciled(idx, update_response._response_info, namespace=target_namespace)

        # Verify the vectors were actually updated
        fetched_after = idx.fetch(ids=["0", "2", "4", "6", "8"], namespace=target_namespace)
        for vec_id in ["0", "2", "4", "6", "8"]:
            assert fetched_after.vectors[vec_id].metadata.get("status") == "updated"
            assert fetched_after.vectors[vec_id].metadata.get("genre") == "comedy"
