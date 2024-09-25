import string
import random
import pytest
from pinecone import PodSpec


def random_string():
    return "".join(random.choice(string.ascii_lowercase) for i in range(10))


class TestCollectionErrorCases:
    def test_create_index_with_nonexistent_source_collection(
        self, client, dimension, metric, environment
    ):
        with pytest.raises(Exception) as e:
            index_name = "from-nonexistent-coll-" + random_string()
            client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=PodSpec(environment=environment, source_collection="doesnotexist"),
            )
            client.delete_index(index_name, -1)
        assert "Resource doesnotexist not found" in str(e.value)

    def test_create_index_in_mismatched_environment(
        self, client, dimension, metric, environment, reusable_collection
    ):
        envs = [
            "eastus-azure",
            "eu-west4-gcp",
            "northamerica-northeast1-gcp",
            "us-central1-gcp",
            "us-west4-gcp",
            "asia-southeast1-gcp",
            "us-east-1-aws",
            "asia-northeast1-gcp",
            "eu-west1-gcp",
            "us-east1-gcp",
            "us-east4-gcp",
            "us-west1-gcp",
        ]
        target_env = random.choice([x for x in envs if x != environment])

        with pytest.raises(Exception) as e:
            index_name = "from-coll-" + random_string()
            client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=PodSpec(environment=target_env, source_collection=reusable_collection),
            )
            client.delete_index(index_name, -1)
        assert "Source collection must be in the same environment as the index" in str(e.value)

    @pytest.mark.skip(reason="Bug reported in #global-cps")
    def test_create_index_with_mismatched_dimension(
        self, client, dimension, metric, environment, reusable_collection
    ):
        with pytest.raises(Exception) as e:
            client.create_index(
                name="from-coll-" + random_string(),
                dimension=dimension + 1,
                metric=metric,
                spec=PodSpec(environment=environment, source_collection=reusable_collection),
            )
        assert "Index and collection must have the same dimension" in str(e.value)

    # def test_create_index_from_notready_collection(self, client, ready_index, random_vector, dimension, metric, environment):
    #     index = client.Index(ready_index)
    #     num_vectors = 10
    #     vectors = [ (str(i), random_vector()) for i in range(num_vectors) ]
    #     index.upsert(vectors=vectors)

    #     collection_name = 'coll-notready-' + random_string()
    #     client.create_collection(name=collection_name, source=ready_index)

    #     # Not doing any waiting for collection to be ready

    #     with pytest.raises(Exception) as e:
    #         client.create_index(
    #             name='coll-notready-idx-' + random_string(),
    #             dimension=dimension,
    #             metric=metric,
    #             spec=PodSpec(
    #                 environment=environment,
    #                 source_collection=collection_name
    #             )
    #         )
    #     client.delete_collection(collection_name)
    #     assert 'Source collection is not ready' in str(e.value)

    def test_create_collection_from_not_ready_index(self, client, notready_index):
        name = "coll3-" + random_string()
        with pytest.raises(Exception) as e:
            client.create_collection(name, notready_index)
        assert "Source index is not ready" in str(e.value)

    def test_create_collection_with_invalid_index(self, client):
        name = "coll4-" + random_string()
        with pytest.raises(Exception) as e:
            client.create_collection(name, "invalid_index")
        assert "Resource invalid_index not found" in str(e.value)
