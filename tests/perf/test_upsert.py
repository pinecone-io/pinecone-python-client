import pytest
import random
import uuid
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC


def create_index(dimension):
    pc = Pinecone()
    index_name = f"perf-index-{random.randint(1,100000)}"
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2"),
    )
    print(f"Created index {index_name}")
    return index_name, pc.describe_index(index_name).host


def cleanup(index_name):
    pc = Pinecone()
    pc.delete_index(name=index_name)
    print(f"Deleted index {index_name}")


def upsert(index, data_to_upsert, batch_size):
    index.upsert(vectors=data_to_upsert, namespace="ns2", batch_size=batch_size)


@pytest.fixture
def dimension():
    return 1024


@pytest.fixture
def batch_size():
    return 10


@pytest.fixture
def num_batches():
    return 100


@pytest.fixture
def data_to_upsert(dimension, batch_size, num_batches):
    vectors = []
    for batch in range(0, num_batches):
        for i in range(0, batch_size):
            vectors.append((str(uuid.uuid4()), [random.random()] * dimension))
    return vectors


class TestUpsert:
    def test_upsert_grpc(self, benchmark, dimension, data_to_upsert, batch_size):
        index_name, host = create_index(dimension)
        index = PineconeGRPC().Index(host=host)

        benchmark.pedantic(upsert, args=(index, data_to_upsert, batch_size), iterations=10)

        cleanup(index_name)

    def test_upsert_rest(self, benchmark, dimension, data_to_upsert, batch_size):
        index_name, host = create_index(dimension)
        index = Pinecone().Index(host=host)

        benchmark.pedantic(upsert, args=(index, data_to_upsert, batch_size), iterations=10)

        cleanup(index_name)
