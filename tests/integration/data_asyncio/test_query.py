import pytest
from pinecone import Vector
from pinecone import PineconeApiException
from .conftest import build_asyncioindex_client, poll_until_lsn_reconciled_async
from ..helpers import random_string, embedding_values

import logging

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
async def test_query(index_host, dimension, target_namespace):
    asyncio_idx = build_asyncioindex_client(index_host)
    logger.info(f"Testing query on index {index_host}")
    logger.info(f"Target namespace: {target_namespace}")
    logger.info(f"Dimension: {dimension}")

    def emb():
        return embedding_values(dimension)

    # Upsert with tuples
    tuple_vectors = [("1", emb()), ("2", emb()), ("3", emb())]
    logger.info(f"Upserting {len(tuple_vectors)} vectors")
    upsert1 = await asyncio_idx.upsert(vectors=tuple_vectors, namespace=target_namespace)

    # Upsert with objects
    object_vectors = [
        Vector(id="4", values=emb(), metadata={"genre": "action"}),
        Vector(id="5", values=emb(), metadata={"genre": "action"}),
        Vector(id="6", values=emb(), metadata={"genre": "horror"}),
    ]
    logger.info(f"Upserting {len(object_vectors)} vectors")
    upsert2 = await asyncio_idx.upsert(vectors=object_vectors, namespace=target_namespace)

    # Upsert with dict
    dict_vectors = [
        {"id": "7", "values": emb()},
        {"id": "8", "values": emb()},
        {"id": "9", "values": emb()},
    ]
    logger.info(f"Upserting {len(dict_vectors)} vectors")
    upsert3 = await asyncio_idx.upsert(vectors=dict_vectors, namespace=target_namespace)

    await poll_until_lsn_reconciled_async(
        asyncio_idx,
        target_lsn=upsert1._response_info.get("lsn_committed"),
        namespace=target_namespace,
    )
    await poll_until_lsn_reconciled_async(
        asyncio_idx,
        target_lsn=upsert2._response_info.get("lsn_committed"),
        namespace=target_namespace,
    )
    await poll_until_lsn_reconciled_async(
        asyncio_idx,
        target_lsn=upsert3._response_info.get("lsn_committed"),
        namespace=target_namespace,
    )

    # Check the vector count reflects some data has been upserted
    stats = await asyncio_idx.describe_index_stats()
    logger.info(f"Index stats: {stats}")
    assert stats.namespaces[target_namespace].vector_count == 9

    results1 = await asyncio_idx.query(top_k=4, namespace=target_namespace, vector=emb())
    logger.info(f"Results 1: {results1}")
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
    logger.info(f"Results 2: {results2}")
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
    logger.info(f"Results 3: {results3}")
    assert results3 is not None
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
    logger.info(f"Results 4: {results4}")
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
    logger.info(f"Results 5: {results5}")
    assert results5 is not None
    assert len(results5.matches) == 0
    assert results5.namespace == target_namespace
    assert results5.usage is not None
    assert results5.usage.read_units is not None
    assert results5.usage.read_units > 0

    # Query by id
    results6 = await asyncio_idx.query(top_k=4, id="1", namespace=target_namespace)
    logger.info(f"Results 6: {results6}")
    assert results6 is not None
    assert len(results6.matches) == 4

    # Query by id, when id doesn't exist gives empty result set
    results7 = await asyncio_idx.query(top_k=10, id="unknown", namespace=target_namespace)
    logger.info(f"Results 7: {results7}")
    assert results7 is not None
    assert len(results7.matches) == 0

    # When missing required top_k kwarg
    with pytest.raises(TypeError) as e:
        await asyncio_idx.query(id="1", namespace=target_namespace)
    logger.info(f"Error Msg 1: {e.value}")
    assert "top_k" in str(e.value)

    # When incorrectly passing top_k as a positional argument
    with pytest.raises(TypeError) as e:
        await asyncio_idx.query(4, id="1", namespace=target_namespace)
    logger.info(f"Error Msg 2: {e.value}")
    assert "top_k" in str(e.value)

    # When trying to pass both id and vector as query params
    with pytest.raises(ValueError) as e:
        await asyncio_idx.query(top_k=10, id="1", vector=emb(), namespace=target_namespace)
    logger.info(f"Error Msg 3: {e.value}")
    assert "Cannot specify both `id` and `vector`" in str(e.value)

    # When trying to pass sparse vector as query params to dense index
    with pytest.raises(PineconeApiException) as e:
        await asyncio_idx.query(
            top_k=10,
            sparse_vector={"indices": [i for i in range(dimension)], "values": emb()},
            namespace=target_namespace,
        )
    logger.info(f"Error Msg 4: {e.value}")
    assert "Cannot query index with dense 'vector_type' with only sparse vector" in str(e.value)
    await asyncio_idx.close()
