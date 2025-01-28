import pytest
from pinecone import Vector
from pinecone import PineconeApiException
from .conftest import build_asyncioindex_client, poll_for_freshness
from ..helpers import random_string, embedding_values


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
async def test_query(pc, index_host, dimension, target_namespace):
    asyncio_idx = build_asyncioindex_client(pc, index_host)

    def emb():
        return embedding_values(dimension)

    # Upsert with tuples
    await asyncio_idx.upsert(
        vectors=[("1", emb()), ("2", emb()), ("3", emb())], namespace=target_namespace
    )

    # Upsert with objects
    await asyncio_idx.upsert(
        vectors=[
            Vector(id="4", values=emb(), metadata={"genre": "action"}),
            Vector(id="5", values=emb(), metadata={"genre": "action"}),
            Vector(id="6", values=emb(), metadata={"genre": "horror"}),
        ],
        namespace=target_namespace,
    )

    # Upsert with dict
    await asyncio_idx.upsert(
        vectors=[
            {"id": "7", "values": emb()},
            {"id": "8", "values": emb()},
            {"id": "9", "values": emb()},
        ],
        namespace=target_namespace,
    )

    await poll_for_freshness(asyncio_idx, target_namespace, 9)

    # Check the vector count reflects some data has been upserted
    stats = await asyncio_idx.describe_index_stats()
    assert stats.total_vector_count >= 9
    # default namespace could have other stuff from other tests
    if target_namespace != "":
        assert stats.namespaces[target_namespace].vector_count == 9

    results1 = await asyncio_idx.query(top_k=4, namespace=target_namespace, vector=emb())
    assert results1 is not None
    assert len(results1.matches) == 4
    assert results1.namespace == target_namespace
    assert results1.matches[0].values == []

    # Response includes usage info
    assert results1.usage is not None
    assert results1.usage.read_units is not None
    assert results1.usage.read_units > 0

    # Response object also supports dictionary-style accessors
    assert results1["matches"] == results1.matches
    assert results1["namespace"] == results1.namespace
    assert results1["matches"][0]["values"] == results1.matches[0].values

    # With include values
    results2 = await asyncio_idx.query(
        top_k=4, namespace=target_namespace, vector=emb(), include_values=True
    )
    assert results2 is not None
    assert len(results2.matches) == 4
    assert results2.namespace == target_namespace
    assert len(results2.matches[0].values) == dimension

    # With filtering
    results3 = await asyncio_idx.query(
        top_k=4,
        namespace=target_namespace,
        vector=emb(),
        filter={"genre": {"$in": ["action"]}},
        include_metadata=True,
        include_values=True,
    )
    assert results3 is not None
    print(results3)
    assert len(results3.matches) == 2
    assert results3.namespace == target_namespace
    assert results3.matches[0].metadata == {"genre": "action"}
    assert results3.matches[1].metadata == {"genre": "action"}
    assert len(results3.matches[0].values) == dimension

    # With filtering, when only one match
    results4 = await asyncio_idx.query(
        top_k=4,
        namespace=target_namespace,
        vector=emb(),
        filter={"genre": {"$in": ["horror"]}},
        include_metadata=True,
        include_values=True,
    )
    assert results4 is not None
    assert len(results4.matches) == 1
    assert results4.namespace == target_namespace
    assert results4.matches[0].metadata == {"genre": "horror"}
    assert len(results4.matches[0].values) == dimension
    assert results4.matches[0].id == "6"

    # With filtering, when no match
    results5 = await asyncio_idx.query(
        top_k=4,
        namespace=target_namespace,
        vector=emb(),
        filter={"genre": {"$in": ["comedy"]}},
        include_metadata=True,
        include_values=True,
    )
    assert results5 is not None
    assert len(results5.matches) == 0
    assert results5.namespace == target_namespace
    assert results5.usage is not None
    assert results5.usage.read_units is not None
    assert results5.usage.read_units > 0

    # Query by id
    results6 = await asyncio_idx.query(top_k=4, id="1", namespace=target_namespace)
    assert results6 is not None
    assert len(results6.matches) == 4

    # Query by id, when id doesn't exist gives empty result set
    results7 = await asyncio_idx.query(top_k=10, id="unknown", namespace=target_namespace)
    assert results7 is not None
    assert len(results7.matches) == 0

    # When missing required top_k kwarg
    with pytest.raises(TypeError) as e:
        await asyncio_idx.query(id="1", namespace=target_namespace)
    assert "top_k" in str(e.value)

    # When incorrectly passing top_k as a positional argument
    with pytest.raises(TypeError) as e:
        await asyncio_idx.query(4, id="1", namespace=target_namespace)
    assert "top_k" in str(e.value)

    # When trying to pass both id and vector as query params
    with pytest.raises(ValueError) as e:
        await asyncio_idx.query(top_k=10, id="1", vector=emb(), namespace=target_namespace)
    assert "Cannot specify both `id` and `vector`" in str(e.value)

    # When trying to pass sparse vector as query params to dense index
    with pytest.raises(PineconeApiException) as e:
        await asyncio_idx.query(
            top_k=10,
            sparse_vector={"indices": [i for i in range(dimension)], "values": emb()},
            namespace=target_namespace,
        )
    assert "Cannot query index with dense 'vector_type' with only sparse vector" in str(e.value)
