import pytest
from pinecone import Pinecone
from tests.integration.helpers import random_string, embedding_values


@pytest.mark.asyncio
async def test_instantiation_through_non_async_client(index_host, dimension):
    asyncio_idx = Pinecone().IndexAsyncio(host=index_host)

    def emb():
        return embedding_values(dimension)

    # Upsert with tuples
    await asyncio_idx.upsert(
        vectors=[("1", emb()), ("2", emb()), ("3", emb())], namespace=random_string(10)
    )

    await asyncio_idx.close()
