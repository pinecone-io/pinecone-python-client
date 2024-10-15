from pinecone.grpc import PineconeGRPC, GRPCClientConfig

# Initialize a client. An API key must be passed, but the
# value does not matter.
pc = PineconeGRPC(api_key="test_api_key")

# Target the indexes. Use the host and port number along with disabling tls.
index1 = pc.Index(host="localhost:5081", grpc_config=GRPCClientConfig(secure=False))
index2 = pc.Index(host="localhost:5082", grpc_config=GRPCClientConfig(secure=False))

# You can now perform data plane operations with index1 and index2

dimension = 3


def upserts():
    vectors = []
    for i in range(0, 100):
        vectors.append((f"vec{i}", [i] * dimension))

    print(len(vectors))

    index1.upsert(vectors=vectors, namespace="ns2")
    index2.upsert(vectors=vectors, namespace="ns2")


upserts()
print(index1.describe_index_stats())

print(index1.query(id="vec1", top_k=2, namespace="ns2", include_values=True))
print(index1.query(id="vec1", top_k=10, namespace="", include_values=True))
