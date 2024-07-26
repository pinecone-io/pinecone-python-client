import string
import random
import pytest
import time
from pinecone import PodSpec


def random_string():
    return "".join(random.choice(string.ascii_lowercase) for i in range(10))


class TestCollectionsHappyPath:
    def test_index_to_collection_to_index_happy_path(
        self, client, environment, dimension, metric, ready_index, random_vector
    ):
        index = client.Index(ready_index)
        num_vectors = 10
        vectors = [(str(i), random_vector()) for i in range(num_vectors)]
        index.upsert(vectors=vectors)

        collection_name = "coll1-" + random_string()
        client.create_collection(name=collection_name, source=ready_index)
        desc = client.describe_collection(collection_name)
        assert desc["name"] == collection_name
        assert desc["environment"] == environment
        assert desc["status"] == "Initializing"

        time_waited = 0
        max_wait = 5 * 60
        collection_ready = desc["status"]
        while collection_ready.lower() != "ready" and time_waited < max_wait:
            print(f"Waiting for collection {collection_name} to be ready. Waited {time_waited} seconds...")
            time.sleep(5)
            time_waited += 5
            desc = client.describe_collection(collection_name)
            collection_ready = desc["status"]

        assert collection_name in client.list_collections().names()

        if time_waited >= max_wait:
            raise Exception(f"Collection {collection_name} is not ready after 5 minutes")

        # After collection ready, these should all be defined
        assert desc["name"] == collection_name
        assert desc["status"] == "Ready"
        assert desc["environment"] == environment
        assert desc["dimension"] == dimension
        assert desc["vector_count"] == num_vectors
        assert desc["size"] != None
        assert desc["size"] > 0

        # Create index from collection
        index_name = "index-from-collection-" + collection_name
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
        client.delete_collection(collection_name)
        client.delete_index(index_name)

    def test_create_index_with_different_metric_from_orig_index(
        self, client, dimension, metric, environment, reusable_collection
    ):
        metrics = ["cosine", "euclidean", "dotproduct"]
        target_metric = random.choice([x for x in metrics if x != metric])

        index_name = "from-coll-" + random_string()
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric=target_metric,
            spec=PodSpec(environment=environment, source_collection=reusable_collection),
        )
        time.sleep(30)
        client.delete_index(index_name, -1)
