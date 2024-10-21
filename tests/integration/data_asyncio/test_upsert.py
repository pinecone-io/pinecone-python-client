import pytest
from pinecone import Vector
from .conftest import use_grpc
from ..helpers import random_string
from .utils import build_asyncio_idx, embedding_values, poll_for_freshness


@pytest.mark.parametrize("target_namespace", ["", random_string(20)])
@pytest.mark.skipif(use_grpc() == False, reason="Currently only GRPC supports asyncio")
async def test_upsert_to_default_namespace(host, dimension, target_namespace):
    asyncio_idx = build_asyncio_idx(host)

    def emb():
        return embedding_values(dimension)

    # Upsert with tuples
    await asyncio_idx.upsert(
        vectors=[("1", emb()), ("2", emb()), ("3", emb())], namespace=target_namespace
    )

    # Upsert with objects
    await asyncio_idx.upsert(
        vectors=[
            Vector(id="4", values=emb()),
            Vector(id="5", values=emb()),
            Vector(id="6", values=emb()),
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

    # # Check the vector count reflects some data has been upserted
    stats = await asyncio_idx.describe_index_stats()
    assert stats.total_vector_count >= 9
    # default namespace could have other stuff from other tests
    if target_namespace != "":
        assert stats.namespaces[target_namespace].vector_count == 9


# @pytest.mark.parametrize("target_namespace", [
#     "",
#     random_string(20),
# ])
# @pytest.mark.skipif(
#     os.getenv("METRIC") != "dotproduct", reason="Only metric=dotprodouct indexes support hybrid"
# )
# async def test_upsert_to_namespace_with_sparse_embedding_values(pc, host, dimension, target_namespace):
#     asyncio_idx = pc.AsyncioIndex(host=host)

#     # Upsert with sparse values object
#     await asyncio_idx.upsert(
#         vectors=[
#             Vector(
#                 id="1",
#                 values=embedding_values(dimension),
#                 sparse_values=SparseValues(indices=[0, 1], values=embedding_values()),
#             )
#         ],
#         namespace=target_namespace,
#     )

#     # Upsert with sparse values dict
#     await asyncio_idx.upsert(
#         vectors=[
#             {
#                 "id": "2",
#                 "values": embedding_values(dimension),
#                 "sparse_values": {"indices": [0, 1], "values": embedding_values()},
#             },
#             {
#                 "id": "3",
#                 "values": embedding_values(dimension),
#                 "sparse_values": {"indices": [0, 1], "values": embedding_values()},
#             },
#         ],
#         namespace=target_namespace,
#     )

#     await poll_for_freshness(asyncio_idx, target_namespace, 9)

#     # Check the vector count reflects some data has been upserted
#     stats = await asyncio_idx.describe_index_stats()
#     assert stats.total_vector_count >= 9

#     if (target_namespace != ""):
#         assert stats.namespaces[target_namespace].vector_count == 9
