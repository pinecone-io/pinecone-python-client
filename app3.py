import asyncio
from pinecone.grpc import PineconeGRPC as Pinecone, Vector

import time
import random
import pandas as pd


# Enable gRPC tracing and verbosity for more detailed logs
# os.environ["GRPC_VERBOSITY"] = "DEBUG"
# os.environ["GRPC_TRACE"] = "all"


# Generate a large set of vectors (as an example)
def generate_vectors(num_vectors, dimension):
    return [
        Vector(id=f"vector_{i}", values=[random.random()] * dimension) for i in range(num_vectors)
    ]


def load_vectors():
    df = pd.read_parquet("test_records_100k_dim1024.parquet")
    df["values"] = df["values"].apply(lambda x: [float(v) for v in x])

    vectors = [Vector(id=row.id, values=list(row.values)) for row in df.itertuples()]
    return vectors


async def main():
    # Create a semaphore to limit concurrency (e.g., max 5 concurrent requests)
    s = time.time()
    # all_vectors = load_vectors()
    all_vectors = generate_vectors(1000, 1024)
    f = time.time()
    print(f"Loaded {len(all_vectors)} vectors in {f-s:.2f} seconds")

    start_time = time.time()

    # Same setup as before...
    pc = Pinecone(api_key="b1cb8ba4-b3d1-458f-9c32-8dd10813459a")
    index = pc.Index(
        # index_host="jen2-dojoi3u.svc.aped-4627-b74a.pinecone.io"
        host="jen1024-dojoi3u.svc.apw5-4e34-81fa.pinecone.io",
        use_asyncio=True,
    )

    batch_size = 150
    namespace = "asyncio-py7"
    res = await index.upsert(
        vectors=all_vectors, batch_size=batch_size, namespace=namespace, show_progress=True
    )

    print(res)

    end_time = time.time()

    total_time = end_time - start_time
    print(f"All tasks completed in {total_time:.2f} seconds")
    print(f"Namespace: {namespace}")


asyncio.run(main())
