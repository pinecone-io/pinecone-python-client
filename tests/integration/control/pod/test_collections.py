import random
import pytest
import time
from pinecone import PodSpec
from ...helpers import generate_index_name, generate_collection_name


def attempt_cleanup_collection(client, collection_name):
    try:
        time.sleep(10)
        client.delete_collection(collection_name)
    except Exception as e:
        # Failures here usually happen because the backend thinks there is still some
        # operation pending on the resource.
        # These orphaned resources will get cleaned up by the cleanup job later.
        print(f"Failed to cleanup collection: {e}")


def attempt_cleanup_index(client, index_name):
    try:
        time.sleep(10)
        client.delete_index(index_name, -1)
    except Exception as e:
        # Failures here usually happen because the backend thinks there is still some
        # operation pending on the resource.
        # These orphaned resources will get cleaned up by the cleanup job later.
        print(f"Failed to cleanup collection: {e}")


class TestCollectionsHappyPath:
    def test_index_to_collection_to_index_happy_path(
        self, client, environment, dimension, metric, ready_index, random_vector
    ):
        index = client.Index(ready_index)
        num_vectors = 10
        vectors = [(str(i), random_vector()) for i in range(num_vectors)]
        index.upsert(vectors=vectors)

        collection_name = generate_collection_name("coll1")
        client.create_collection(name=collection_name, source=ready_index)
        desc = client.describe_collection(collection_name)
        assert desc["name"] == collection_name
        assert desc["environment"] == environment
        assert desc["status"] == "Initializing"

        time_waited = 0
        collection_ready = desc["status"]
        while collection_ready.lower() != "ready" and time_waited < 120:
            print(
                f"Waiting for collection {collection_name} to be ready. Waited {time_waited} seconds..."
            )
            time.sleep(5)
            time_waited += 5
            desc = client.describe_collection(collection_name)
            collection_ready = desc["status"]

        assert collection_name in client.list_collections().names()

        if time_waited >= 120:
            raise Exception(f"Collection {collection_name} is not ready after 120 seconds")

        # After collection ready, these should all be defined
        assert desc["name"] == collection_name
        assert desc["status"] == "Ready"
        assert desc["environment"] == environment
        assert desc["dimension"] == dimension
        assert desc["vector_count"] == num_vectors
        assert desc["size"] is not None
        assert desc["size"] > 0

        # Create index from collection
        index_name = generate_index_name("index-from-collection-" + collection_name)
        print(f"Creating index {index_name} from collection {collection_name}...")
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=PodSpec(environment=environment, source_collection=collection_name),
        )
        print(
            f"Created index {index_name} from collection {collection_name}. Waiting a little more to make sure it's ready..."
        )
        time.sleep(30)
        desc = client.describe_index(index_name)
        assert desc["name"] == index_name
        assert desc["status"]["ready"] == True

        new_index = client.Index(index_name)

        # Verify stats reflect the vectors present in the collection
        stats = new_index.describe_index_stats()
        print(stats)
        assert stats.total_vector_count == num_vectors

        # Verify the vectors from the collection can be fetched
        results = new_index.fetch(ids=[v[0] for v in vectors])
        print(results)
        for v in vectors:
            assert results.vectors[v[0]].id == v[0]
            assert results.vectors[v[0]].values == pytest.approx(v[1], rel=0.01)

        # Cleanup
        attempt_cleanup_collection(client, collection_name)
        attempt_cleanup_index(client, index_name)

    def test_create_index_with_different_metric_from_orig_index(
        self, client, dimension, metric, environment, reusable_collection
    ):
        metrics = ["cosine", "euclidean", "dotproduct"]
        target_metric = random.choice([x for x in metrics if x != metric])

        index_name = generate_index_name("from-" + reusable_collection)
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric=target_metric,
            spec=PodSpec(environment=environment, source_collection=reusable_collection),
        )
        attempt_cleanup_index(client, index_name)
