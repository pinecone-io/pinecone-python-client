from pinecone.grpc import PineconeGRPC
from pinecone import Pinecone

pc = Pinecone(api_key="b1cb8ba4-b3d1-458f-9c32-8dd10813459a")
pcg = PineconeGRPC(api_key="b1cb8ba4-b3d1-458f-9c32-8dd10813459a")

index = pc.Index("jen2")
indexg = pcg.Index(name="jen2", use_asyncio=True)

# Rest call fails
# print(index.upsert(vectors=[("vec1", [1, 2])]))

# GRPC succeeds
print(indexg.upsert(vectors=[("vec1", [1, 2])]))

# print(index.fetch(ids=['vec1']))
