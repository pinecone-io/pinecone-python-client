import pytest
import time
from pinecone import Vector
from ..helpers import poll_fetch_for_ids_in_namespace, embedding_values, random_string


@pytest.fixture(scope="session")
def update_namespace():
    return random_string(10)


def seed_for_update(idx, namespace):
    """Seed test data for update tests."""
    logger = __import__("logging").getLogger(__name__)
    logger.info(f"Seeding vectors for update tests in namespace '{namespace}'")
    idx.upsert(
        vectors=[
            Vector(
                id=str(i),
                values=embedding_values(2),
                metadata={"genre": "action", "year": 2020, "status": "active"},
            )
            for i in range(10)
        ],
        namespace=namespace,
    )
    poll_fetch_for_ids_in_namespace(idx, ids=[str(i) for i in range(10)], namespace=namespace)


@pytest.fixture(scope="class")
def seed_for_update_tests(idx, update_namespace):
    seed_for_update(idx, update_namespace)
    seed_for_update(idx, "")
    yield


def poll_until_update_reflected(
    idx, vector_id, namespace, expected_values=None, expected_metadata=None, timeout=180
):
    """Poll fetch until update is reflected in the vector."""
    logger = __import__("logging").getLogger(__name__)
    delta_t = 2  # Start with shorter interval
    total_time = 0
    max_delta_t = 10  # Max interval

    while total_time < timeout:
        logger.debug(
            f'Polling for update on vector "{vector_id}" in namespace "{namespace}". Total time waited: {total_time} seconds'
        )
        try:
            results = idx.fetch(ids=[vector_id], namespace=namespace)
            if vector_id in results.vectors:
                vec = results.vectors[vector_id]

                # If both are None, we just check that the vector exists
                if expected_values is None and expected_metadata is None:
                    return  # Vector exists, we're done

                values_match = True
                metadata_match = True

                if expected_values is not None:
                    if vec.values is None:
                        values_match = False
                    else:
                        if len(vec.values) != len(expected_values):
                            values_match = False
                        else:
                            values_match = all(
                                vec.values[i] == pytest.approx(expected_values[i], 0.01)
                                for i in range(len(expected_values))
                            )

                if expected_metadata is not None:
                    # Check that all expected metadata fields are present and match
                    # (metadata may be merged, so we check for our fields specifically)
                    if vec.metadata is None:
                        metadata_match = False
                    else:
                        metadata_match = all(
                            vec.metadata.get(k) == v for k, v in expected_metadata.items()
                        )

                if values_match and metadata_match:
                    logger.debug(f"Update reflected for vector {vector_id}")
                    return  # Update is reflected
        except Exception as e:
            logger.debug(f"Error while polling: {e}")

        time.sleep(delta_t)
        total_time += delta_t
        # Gradually increase interval up to max
        delta_t = min(delta_t * 1.5, max_delta_t)

    raise TimeoutError(
        f"Timed out waiting for update on vector {vector_id} in namespace {namespace} after {total_time} seconds"
    )


@pytest.mark.usefixtures("seed_for_update_tests")
class TestUpdate:
    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    def test_update_values(self, idx, update_namespace, use_nondefault_namespace):
        """Test updating vector values by ID."""
        target_namespace = update_namespace if use_nondefault_namespace else ""
        vector_id = "1"

        # Update values
        new_values = embedding_values(2)
        idx.update(id=vector_id, values=new_values, namespace=target_namespace)

        # Wait for update to be reflected
        poll_until_update_reflected(
            idx, vector_id, target_namespace, expected_values=new_values, timeout=180
        )

        # Verify the update
        fetched_vec = idx.fetch(ids=[vector_id], namespace=target_namespace)
        assert fetched_vec.vectors[vector_id].values[0] == pytest.approx(new_values[0], 0.01)
        assert fetched_vec.vectors[vector_id].values[1] == pytest.approx(new_values[1], 0.01)

    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    def test_update_metadata(self, idx, update_namespace, use_nondefault_namespace):
        """Test updating vector metadata by ID."""
        target_namespace = update_namespace if use_nondefault_namespace else ""
        vector_id = "2"

        # Update metadata
        new_metadata = {"genre": "comedy", "year": 2021, "status": "inactive"}
        idx.update(id=vector_id, set_metadata=new_metadata, namespace=target_namespace)

        # Wait for update to be reflected
        poll_until_update_reflected(
            idx, vector_id, target_namespace, expected_metadata=new_metadata, timeout=180
        )

        # Verify the update
        fetched_vec = idx.fetch(ids=[vector_id], namespace=target_namespace)
        assert fetched_vec.vectors[vector_id].metadata == new_metadata

    @pytest.mark.parametrize("use_nondefault_namespace", [True, False])
    def test_update_values_and_metadata(self, idx, update_namespace, use_nondefault_namespace):
        """Test updating both vector values and metadata by ID."""
        target_namespace = update_namespace if use_nondefault_namespace else ""
        vector_id = "3"

        # Update both values and metadata
        new_values = embedding_values(2)
        new_metadata = {"genre": "drama", "year": 2022, "status": "pending"}
        idx.update(
            id=vector_id, values=new_values, set_metadata=new_metadata, namespace=target_namespace
        )

        # Wait for update to be reflected
        poll_until_update_reflected(
            idx,
            vector_id,
            target_namespace,
            expected_values=new_values,
            expected_metadata=new_metadata,
            timeout=180,
        )

        # Verify the update
        fetched_vec = idx.fetch(ids=[vector_id], namespace=target_namespace)
        assert fetched_vec.vectors[vector_id].values[0] == pytest.approx(new_values[0], 0.01)
        assert fetched_vec.vectors[vector_id].values[1] == pytest.approx(new_values[1], 0.01)
        assert fetched_vec.vectors[vector_id].metadata == new_metadata

    def test_update_only_metadata_no_values(self, idx, update_namespace):
        """Test updating only metadata without providing values."""
        target_namespace = update_namespace
        vector_id = "4"

        # Get original values first
        original_vec = idx.fetch(ids=[vector_id], namespace=target_namespace)
        original_values = original_vec.vectors[vector_id].values

        # Update only metadata
        new_metadata = {"genre": "thriller", "year": 2023}
        idx.update(id=vector_id, set_metadata=new_metadata, namespace=target_namespace)

        # Wait for update to be reflected - check that specified fields are present
        # Note: set_metadata may replace or merge, so we check for the fields we set
        def check_metadata_update():
            fetched = idx.fetch(ids=[vector_id], namespace=target_namespace)
            if vector_id in fetched.vectors:
                vec = fetched.vectors[vector_id]
                if vec.metadata is not None:
                    # Check that our specified fields match
                    return (
                        vec.metadata.get("genre") == "thriller" and vec.metadata.get("year") == 2023
                    )
            return False

        timeout = 180
        delta_t = 2
        total_time = 0
        max_delta_t = 10

        while total_time < timeout:
            if check_metadata_update():
                break
            time.sleep(delta_t)
            total_time += delta_t
            delta_t = min(delta_t * 1.5, max_delta_t)
        else:
            raise TimeoutError(
                f"Timed out waiting for metadata update on vector {vector_id} in namespace {target_namespace}"
            )

        # Verify metadata updated but values unchanged
        fetched_vec = idx.fetch(ids=[vector_id], namespace=target_namespace)
        # Check that the fields we set are present
        assert fetched_vec.vectors[vector_id].metadata is not None
        assert fetched_vec.vectors[vector_id].metadata.get("genre") == "thriller"
        assert fetched_vec.vectors[vector_id].metadata.get("year") == 2023
        # Values should remain the same (approximately, due to floating point)
        assert len(fetched_vec.vectors[vector_id].values) == len(original_values)
        assert fetched_vec.vectors[vector_id].values[0] == pytest.approx(original_values[0], 0.01)
