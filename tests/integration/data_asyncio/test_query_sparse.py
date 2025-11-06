import pytest
import random
from pinecone import Vector, SparseValues, PineconeApiException
from .conftest import build_asyncioindex_client, poll_until_lsn_reconciled_async
from ..helpers import random_string, embedding_values


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
async def test_query_sparse(sparse_index_host, target_namespace):
    asyncio_sparse_idx = build_asyncioindex_client(sparse_index_host)

    # Upsert with Vector objects containing sparse values dict
    upsert1 = await asyncio_sparse_idx.upsert(
        vectors=[
            Vector(
                id=str(i),
                sparse_values={"indices": [j for j in range(100)], "values": embedding_values(100)},
                metadata={"genre": "action", "runtime": random.randint(90, 180)},
            )
            for i in range(50)
        ],
        namespace=target_namespace,
    )
    # Make one have unique metadata for later assertions
    upsert2 = await asyncio_sparse_idx.upsert(
        vectors=[
            Vector(
                id=str(10),
                sparse_values={"indices": [j for j in range(100)], "values": embedding_values(100)},
                metadata={"genre": "documentary", "runtime": random.randint(90, 180)},
            )
        ],
        namespace=target_namespace,
    )

    # Upsert with objects with SparseValues object
    upsert3 = await asyncio_sparse_idx.upsert(
        vectors=[
            Vector(
                id=str(i),
                sparse_values=SparseValues(
                    indices=[j for j in range(100)], values=embedding_values(100)
                ),
                metadata={"genre": "horror", "runtime": random.randint(90, 180)},
            )
            for i in range(50, 100)
        ],
        namespace=target_namespace,
    )

    # Upsert with dict
    upsert4 = await asyncio_sparse_idx.upsert(
        vectors=[
            {
                "id": str(i),
                "sparse_values": {
                    "indices": [i, random.randint(2000, 4000)],
                    "values": embedding_values(2),
                },
                "metadata": {"genre": "comedy", "runtime": random.randint(90, 180)},
            }
            for i in range(100, 151)
        ],
        namespace=target_namespace,
    )

    # Upsert with mixed types, dict with SparseValues object
    upsert5 = await asyncio_sparse_idx.upsert(
        vectors=[
            {
                "id": str(i),
                "sparse_values": SparseValues(
                    indices=[i, random.randint(2000, 4000)], values=embedding_values(2)
                ),
            }
            for i in range(150, 200)
        ],
        namespace=target_namespace,
    )

    await poll_until_lsn_reconciled_async(
        asyncio_sparse_idx,
        target_lsn=upsert1._response_info.get("lsn_committed"),
        namespace=target_namespace,
    )
    await poll_until_lsn_reconciled_async(
        asyncio_sparse_idx,
        target_lsn=upsert2._response_info.get("lsn_committed"),
        namespace=target_namespace,
    )
    await poll_until_lsn_reconciled_async(
        asyncio_sparse_idx,
        target_lsn=upsert3._response_info.get("lsn_committed"),
        namespace=target_namespace,
    )
    await poll_until_lsn_reconciled_async(
        asyncio_sparse_idx,
        target_lsn=upsert4._response_info.get("lsn_committed"),
        namespace=target_namespace,
    )
    await poll_until_lsn_reconciled_async(
        asyncio_sparse_idx,
        target_lsn=upsert5._response_info.get("lsn_committed"),
        namespace=target_namespace,
    )

    # # Check the vector count reflects some data has been upserted
    stats = await asyncio_sparse_idx.describe_index_stats()
    assert stats.total_vector_count >= 200
    # default namespace could have other stuff from other tests
    if target_namespace != "":
        assert stats.namespaces[target_namespace].vector_count == 200

    results1 = await asyncio_sparse_idx.query(
        top_k=4,
        namespace=target_namespace,
        sparse_vector=SparseValues(indices=[1, 2, 3, 4, 5], values=embedding_values(5)),
    )
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
    assert results1["matches"][0]["id"] == results1.matches[0].id

    # With include values
    results2 = await asyncio_sparse_idx.query(
        top_k=4,
        namespace=target_namespace,
        sparse_vector=SparseValues(indices=[1, 2, 3, 4, 5], values=embedding_values(5)),
        include_values=True,
    )
    assert results2 is not None
    assert len(results2.matches) == 4
    assert results2.namespace == target_namespace
    assert results2.matches[0].sparse_values is not None
    assert len(results2.matches[0].sparse_values.indices) > 0
    assert len(results2.matches[0].sparse_values.values) > 0
    assert len(results2.matches[0].values) == 0

    # With filtering
    results3 = await asyncio_sparse_idx.query(
        top_k=4,
        namespace=target_namespace,
        sparse_vector=SparseValues(indices=[1, 2, 3, 4, 5], values=embedding_values(5)),
        filter={"genre": {"$in": ["action"]}},
        include_metadata=True,
        include_values=True,
    )
    assert results3 is not None
    assert len(results3.matches) == 4
    assert results3.namespace == target_namespace
    assert results3.matches[0].metadata["genre"] == "action"

    # With filtering, when only one match
    results4 = await asyncio_sparse_idx.query(
        top_k=4,
        namespace=target_namespace,
        sparse_vector=SparseValues(indices=[1, 2, 3, 4, 5], values=embedding_values(5)),
        filter={"genre": {"$in": ["documentary"]}},
        include_metadata=True,
        include_values=True,
    )
    assert results4 is not None
    assert len(results4.matches) == 1
    assert results4.namespace == target_namespace
    assert results4.matches[0].metadata["genre"] == "documentary"
    assert results4.matches[0].id == "10"

    # With filtering, when no match
    results5 = await asyncio_sparse_idx.query(
        top_k=4,
        namespace=target_namespace,
        sparse_vector=SparseValues(indices=[1, 2, 3, 4, 5], values=embedding_values(5)),
        filter={"genre": {"$in": ["thriller"]}},
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
    results6 = await asyncio_sparse_idx.query(top_k=4, id="1", namespace=target_namespace)
    assert results6 is not None
    assert len(results6.matches) == 4

    # Query by id, when id doesn't exist gives empty result set
    results7 = await asyncio_sparse_idx.query(top_k=10, id="unknown", namespace=target_namespace)
    assert results7 is not None
    assert len(results7.matches) == 0

    # When missing required top_k kwarg
    with pytest.raises(TypeError) as e:
        await asyncio_sparse_idx.query(id="1", namespace=target_namespace)
    assert "top_k" in str(e.value)

    # When incorrectly passing top_k as a positional argument
    with pytest.raises(TypeError) as e:
        await asyncio_sparse_idx.query(4, id="1", namespace=target_namespace)
    assert "top_k" in str(e.value)

    # When trying to pass both id and sparse_vector as query params
    with pytest.raises(PineconeApiException) as e:
        await asyncio_sparse_idx.query(
            top_k=10,
            id="1",
            sparse_vector=SparseValues(indices=[1, 2, 3, 4, 5], values=embedding_values(5)),
            namespace=target_namespace,
        )
    assert "Cannot provide both 'ID' and 'sparse_vector' at the same time" in str(e.value)

    # When trying to query with dense vector on sparse index
    with pytest.raises(PineconeApiException) as e:
        await asyncio_sparse_idx.query(
            top_k=10, vector=embedding_values(10), namespace=target_namespace
        )
    assert "Either 'sparse_vector' or 'ID' must be provided" in str(e.value)

    # When trying to pass dense vector as query params to sparse index
    with pytest.raises(PineconeApiException) as e:
        await asyncio_sparse_idx.query(
            top_k=10,
            sparse_vector=SparseValues(indices=[1, 2, 3, 4, 5], values=embedding_values(5)),
            vector=embedding_values(10),
            namespace=target_namespace,
        )
    assert "Index configuration does not support dense values" in str(e.value)
    await asyncio_sparse_idx.close()
