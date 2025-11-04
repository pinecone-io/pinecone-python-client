import pytest
import time
from pinecone import Vector
from ..helpers import poll_stats_for_namespace, embedding_values, generate_name


@pytest.fixture(scope="class")
def namespace_update_async(request):
    return generate_name(request.node.name, "update-namespace")


def seed_for_update_async(idx, namespace):
    """Seed test data for async update tests."""
    logger = __import__("logging").getLogger(__name__)
    logger.info(f"Seeding vectors for async update tests in namespace '{namespace}'")
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
    poll_stats_for_namespace(idx, namespace, 10)


@pytest.fixture(scope="class")
def seed_for_update_async_tests(idx, namespace_update_async):
    seed_for_update_async(idx, namespace_update_async)
    yield


def poll_until_update_reflected_async(
    idx, vector_id, namespace, expected_values=None, expected_metadata=None, timeout=180
):
    """Poll fetch until update is reflected in the vector (for async updates)."""
    logger = __import__("logging").getLogger(__name__)
    delta_t = 2  # Start with shorter interval
    total_time = 0
    max_delta_t = 10  # Max interval

    while total_time < timeout:
        logger.debug(
            f'Polling for async update on vector "{vector_id}" in namespace "{namespace}". Total time waited: {total_time} seconds'
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
        f"Timed out waiting for async update on vector {vector_id} in namespace {namespace} after {total_time} seconds"
    )


@pytest.mark.usefixtures("seed_for_update_async_tests")
class TestUpdateWithAsyncReq:
    def test_update_values_async(self, idx, namespace_update_async):
        """Test updating vector values by ID with async_req=True."""
        target_namespace = namespace_update_async
        vector_id = "1"

        # Update values with async request
        new_values = embedding_values(2)
        future = idx.update(
            id=vector_id, values=new_values, namespace=target_namespace, async_req=True
        )

        # Wait for future to complete
        result = future.result()
        assert result == {}  # Update response should be empty dict

        # Wait for update to be reflected
        poll_until_update_reflected_async(
            idx, vector_id, target_namespace, expected_values=new_values, timeout=180
        )

        # Verify the update
        fetched_vec = idx.fetch(ids=[vector_id], namespace=target_namespace)
        assert fetched_vec.vectors[vector_id].values[0] == pytest.approx(new_values[0], 0.01)
        assert fetched_vec.vectors[vector_id].values[1] == pytest.approx(new_values[1], 0.01)

    def test_update_metadata_async(self, idx, namespace_update_async):
        """Test updating vector metadata by ID with async_req=True."""
        target_namespace = namespace_update_async
        vector_id = "2"

        # Update metadata with async request
        new_metadata = {"genre": "comedy", "year": 2021, "status": "inactive"}
        future = idx.update(
            id=vector_id, set_metadata=new_metadata, namespace=target_namespace, async_req=True
        )

        # Wait for future to complete
        result = future.result()
        assert result == {}  # Update response should be empty dict

        # Wait for update to be reflected
        poll_until_update_reflected_async(
            idx, vector_id, target_namespace, expected_metadata=new_metadata, timeout=180
        )

        # Verify the update
        fetched_vec = idx.fetch(ids=[vector_id], namespace=target_namespace)
        assert fetched_vec.vectors[vector_id].metadata == new_metadata

    def test_update_values_and_metadata_async(self, idx, namespace_update_async):
        """Test updating both vector values and metadata by ID with async_req=True."""
        target_namespace = namespace_update_async
        vector_id = "3"

        # Update both values and metadata with async request
        new_values = embedding_values(2)
        new_metadata = {"genre": "drama", "year": 2022, "status": "pending"}
        future = idx.update(
            id=vector_id,
            values=new_values,
            set_metadata=new_metadata,
            namespace=target_namespace,
            async_req=True,
        )

        # Wait for future to complete
        result = future.result()
        assert result == {}  # Update response should be empty dict

        # Wait for update to be reflected
        poll_until_update_reflected_async(
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

    def test_update_multiple_async(self, idx, namespace_update_async):
        """Test updating multiple vectors asynchronously."""
        target_namespace = namespace_update_async

        # Update multiple vectors with async requests
        futures = []
        updates = []
        for i in range(5, 8):
            new_values = embedding_values(2)
            new_metadata = {"genre": f"genre_{i}", "updated": True}
            future = idx.update(
                id=str(i),
                values=new_values,
                set_metadata=new_metadata,
                namespace=target_namespace,
                async_req=True,
            )
            futures.append(future)
            updates.append((str(i), new_values, new_metadata))

        # Wait for all futures to complete
        for future in futures:
            result = future.result()
            assert result == {}  # Update response should be empty dict

        # Wait for all updates to be reflected - check each one individually
        # with a reasonable timeout per vector
        for vector_id, new_values, new_metadata in updates:
            poll_until_update_reflected_async(
                idx,
                vector_id,
                target_namespace,
                expected_values=new_values,
                expected_metadata=new_metadata,
                timeout=240,  # Increased timeout for async operations
            )

        # Verify all updates
        fetched_vecs = idx.fetch(ids=[str(i) for i in range(5, 8)], namespace=target_namespace)
        for vector_id, new_values, new_metadata in updates:
            assert fetched_vecs.vectors[vector_id].values[0] == pytest.approx(new_values[0], 0.01)
            # Check that metadata fields are present (may be merged with existing)
            assert fetched_vecs.vectors[vector_id].metadata is not None
            assert fetched_vecs.vectors[vector_id].metadata.get("genre") == new_metadata["genre"]
            assert (
                fetched_vecs.vectors[vector_id].metadata.get("updated") == new_metadata["updated"]
            )
