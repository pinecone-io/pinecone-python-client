import random
import uuid
from pinecone.grpc import PineconeGRPC, GRPCClientConfig

# Initialize a client. An API key must be passed, but the 
# value does not matter.
pc = PineconeGRPC(api_key="test_api_key")

# Target the indexes. Use the host and port number along with disabling tls.
index = pc.Index(host="localhost:5081", grpc_config=GRPCClientConfig(secure=False))
dimension = 3

def upserts():
    vectors = []
    for batch in range(0, 10):
        for i in range(0, 100):
            vectors.append((str(uuid.uuid4()), [random.random()] * dimension))
        index.upsert(vectors=vectors, namespace="ns2")

def test_upsert(benchmark):
    benchmark(upserts)
    print(index.describe_index_stats())
