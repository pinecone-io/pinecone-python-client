import pytest
import random
import time
from pinecone import Pinecone, PodSpec, PodIndexEnvironment
from pinecone.exceptions import NotFoundException
from tests.integration.helpers import generate_index_name, generate_collection_name


@pytest.fixture()
def client():
    return Pinecone()


@pytest.fixture()
def environment():
    return PodIndexEnvironment.US_EAST1_AWS.value


@pytest.fixture()
def dimension():
    return 2


@pytest.fixture()
def create_index_params(index_name, environment, dimension, metric):
    spec = {"pod": {"environment": environment, "pod_type": "p1.x1"}}
    return dict(name=index_name, dimension=dimension, metric=metric, spec=spec, timeout=-1)


@pytest.fixture()
def metric():
    return "cosine"


@pytest.fixture()
def random_vector(dimension):
    def _random_vector():
        return [random.uniform(0, 1) for _ in range(dimension)]

    return _random_vector


@pytest.fixture()
def index_name(request):
    test_name = request.node.name
    return generate_index_name(test_name)


@pytest.fixture()
def ready_index(client, index_name, create_index_params):
    create_index_params["timeout"] = None
    client.create_index(**create_index_params)
    time.sleep(10)  # Extra wait, since status is sometimes inaccurate
    yield index_name
    attempt_delete_index(client, index_name)


@pytest.fixture()
def notready_index(client, index_name, create_index_params):
    create_index_params.update({"timeout": -1})
    client.create_index(**create_index_params)
    yield index_name


@pytest.fixture(scope="session")
def reusable_collection():
    pc = Pinecone()
    index_name = generate_index_name("temp-index")
    dimension = 2
    print(f"Creating index {index_name} to prepare a collection...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=PodSpec(environment=PodIndexEnvironment.US_EAST1_AWS.value),
    )
    print(f"Created index {index_name}. Waiting 10 seconds to make sure it's ready...")
    time.sleep(10)

    num_vectors = 10
    vectors = [
        (str(i), [random.uniform(0, 1) for _ in range(dimension)]) for i in range(num_vectors)
    ]

    index = pc.Index(index_name)
    index.upsert(vectors=vectors)

    collection_name = generate_collection_name("reused-coll")
    pc.create_collection(name=collection_name, source=index_name)

    time_waited = 0
    desc = pc.describe_collection(collection_name)
    collection_ready = desc["status"]
    while collection_ready.lower() != "ready" and time_waited < 120:
        print(
            f"Waiting for collection {collection_name} to be ready. Waited {time_waited} seconds..."
        )
        time.sleep(5)
        time_waited += 5
        desc = pc.describe_collection(collection_name)
        collection_ready = desc["status"]

    if time_waited >= 120:
        raise Exception(f"Collection {collection_name} is not ready after 120 seconds")

    print(f"Collection {collection_name} is ready. Deleting index {index_name}...")
    attempt_delete_index(pc, index_name)

    yield collection_name

    print(f"Deleting collection {collection_name}...")
    attempt_delete_collection(pc, collection_name)


def attempt_delete_collection(client, collection_name):
    time_waited = 0
    while collection_name in client.list_collections().names() and time_waited < 120:
        print(
            f"Waiting for collection {collection_name} to be ready to delete. Waited {time_waited} seconds.."
        )
        time_waited += 5
        time.sleep(5)
        try:
            print(f"Attempting delete of collection {collection_name}")
            client.delete_collection(collection_name)
            print(f"Deleted collection {collection_name}")
            break
        except Exception as e:
            print(f"Unable to delete collection {collection_name}: {e}")
            pass

    if time_waited >= 120:
        # Things that fail to delete due to transient statuses will be garbage
        # collected by the nightly cleanup script
        print(f"Collection {collection_name} could not be deleted after 120 seconds")


def attempt_delete_index(client, index_name):
    time_waited = 0
    while client.has_index(index_name) and time_waited < 120:
        try:
            if client.describe_index(index_name).delete_protection == "enabled":
                client.configure_index(index_name, deletion_protection="disabled")
        except NotFoundException:
            # Index was deleted between has_index check and describe_index call
            # Exit the loop since the index no longer exists
            break

        print(
            f"Waiting for index {index_name} to be ready to delete. Waited {time_waited} seconds.."
        )
        time_waited += 5
        time.sleep(5)
        try:
            print(f"Attempting delete of index {index_name}")
            client.delete_index(index_name, -1)
            print(f"Deleted index {index_name}")
            break
        except Exception as e:
            print(f"Unable to delete index {index_name}: {e}")
            pass

    if time_waited >= 120:
        # Things that fail to delete due to transient statuses will be garbage
        # collected by the nightly cleanup script
        print(f"Index {index_name} could not be deleted after 120 seconds")


@pytest.fixture(autouse=True)
def cleanup(client, index_name):
    yield

    attempt_delete_index(client, index_name)
