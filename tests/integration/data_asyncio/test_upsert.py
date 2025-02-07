import pytest
from pinecone import Vector, PineconeApiException, PineconeApiTypeError
from .conftest import build_asyncioindex_client
from ..helpers import random_string, embedding_values


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
async def test_upsert_with_batch_size_dense(index_host, dimension, target_namespace):
    asyncio_idx = build_asyncioindex_client(index_host)

    await asyncio_idx.upsert(
        vectors=[Vector(id=str(i), values=embedding_values(dimension)) for i in range(100)],
        namespace=target_namespace,
        batch_size=10,
        show_progress=False,
    )
    await asyncio_idx.close()


@pytest.mark.asyncio
async def test_upsert_dense_errors(index_host, dimension):
    asyncio_idx = build_asyncioindex_client(index_host)

    # When upserting vectors with incorrect dimension
    with pytest.raises(PineconeApiException) as e:
        await asyncio_idx.upsert(
            vectors=[Vector(id="1", values=embedding_values(dimension + 1))], show_progress=False
        )
    assert "Vector dimension 3 does not match the dimension of the index" in str(e.value)

    # When upserting vectors with incorrect dimension in tuple
    with pytest.raises(PineconeApiException) as e:
        await asyncio_idx.upsert(
            vectors=[("1", embedding_values(dimension + 1))], show_progress=False
        )
    assert "Vector dimension 3 does not match the dimension of the index" in str(e.value)

    # When upserting vectors with incorrect dimension in dict
    with pytest.raises(PineconeApiException) as e:
        await asyncio_idx.upsert(
            vectors=[{"id": "1", "values": embedding_values(dimension + 1)}], show_progress=False
        )
    assert "Vector dimension 3 does not match the dimension of the index" in str(e.value)

    # When upserting vectors with missing id
    with pytest.raises(TypeError) as e:
        await asyncio_idx.upsert(
            vectors=[Vector(id=None, values=embedding_values(dimension))], show_progress=False
        )
    assert "missing 1 required positional argument" in str(e.value)

    # When upserting with missing id from tuple
    with pytest.raises(PineconeApiTypeError) as e:
        await asyncio_idx.upsert(vectors=[(None, embedding_values(dimension))], show_progress=False)
    assert "Invalid type for variable 'id'" in str(e.value)

    # When upserting with empty vector array
    with pytest.raises(PineconeApiException) as e:
        await asyncio_idx.upsert(vectors=[])
    assert "Invalid request" in str(e.value)
    await asyncio_idx.close()
