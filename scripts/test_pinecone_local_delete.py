"""One-off repro script for GitHub issue #564.

Requires pinecone-local running in Docker:

    docker run -it --rm \
      --name pinecone \
      --platform linux/amd64 \
      -e PORT=15080 \
      -e PINECONE_HOST=localhost \
      -p 15080-15090:15080-15090 \
      ghcr.io/pinecone-io/pinecone-local:latest

Then run:

    uv run python scripts/test_pinecone_local_delete.py
"""

import asyncio

import numpy as np
import pinecone


async def main():
    index_name = "test-pinecone-index"
    pc = pinecone.Pinecone(api_key="test-api-key", host="http://localhost:15080")

    if pc.has_index(index_name):
        pc.delete_index(index_name)

    pc.create_index(
        name=index_name,
        dimension=512,
        metric="cosine",
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection="disabled",
    )

    index_host = f"http://{pc.describe_index(index_name)['host']}"
    idx = pc.IndexAsyncio(host=index_host)

    try:
        await idx.upsert(
            vectors=[
                ("vec1", np.random.rand(512).astype(np.float32).tolist()),
                ("vec2", np.random.rand(512).astype(np.float32).tolist()),
            ],
            namespace="test-namespace",
        )

        # This is the call that failed in issue #564
        await idx.delete(namespace="test-namespace", delete_all=True)
        print("PASS: async delete_all succeeded")

        # Also test delete by ids
        await idx.upsert(
            vectors=[
                ("vec3", np.random.rand(512).astype(np.float32).tolist()),
            ],
            namespace="test-namespace",
        )
        await idx.delete(ids=["vec3"], namespace="test-namespace")
        print("PASS: async delete by ids succeeded")

        # Test sync client too
        sync_idx = pc.Index(host=index_host)
        sync_idx.upsert(
            vectors=[
                ("vec4", np.random.rand(512).astype(np.float32).tolist()),
            ],
            namespace="test-namespace",
        )
        sync_idx.delete(namespace="test-namespace", delete_all=True)
        print("PASS: sync delete_all succeeded")
    finally:
        await idx.close()
        pc.delete_index(index_name)

    print("\nAll checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
