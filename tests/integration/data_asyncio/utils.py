import random
import asyncio
from pinecone.grpc import PineconeGRPC as Pinecone


def build_asyncio_idx(host):
    return Pinecone().AsyncioIndex(host=host)


def embedding_values(dimension=2):
    return [random.random() for _ in range(dimension)]


async def poll_for_freshness(asyncio_idx, namespace, expected_count):
    total_wait = 0
    delta = 2
    while True:
        stats = await asyncio_idx.describe_index_stats()
        if stats.namespaces.get(namespace, None) is not None:
            if stats.namespaces[namespace].vector_count >= expected_count:
                print(
                    f"Found {stats.namespaces[namespace].vector_count} vectors in namespace '{namespace}' after {total_wait} seconds"
                )
                break
        await asyncio.sleep(delta)
        total_wait += delta

        if total_wait > 60:
            raise TimeoutError(
                f"Timed out waiting for vectors to appear in namespace '{namespace}'"
            )
