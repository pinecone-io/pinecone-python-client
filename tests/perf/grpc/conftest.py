import pytest
import time

@pytest.fixture(scope="session")
def idx():
    from pinecone import ServerlessSpec
    from pinecone.grpc import PineconeGRPC
    pc = PineconeGRPC()

    index_name = 'perf-testing'
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name, 
            dimension=2,
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )

    return pc.Index(index_name)

def setup_list_data(idx, target_namespace, wait):
    # Upsert a bunch more stuff for testing list pagination
    for i in range(0, 1000, 50):
        idx.upsert(vectors=[
                (str(i+d), [0.111, 0.222]) for d in range(50)
            ],
            namespace=target_namespace
        )
    time.sleep(60)
