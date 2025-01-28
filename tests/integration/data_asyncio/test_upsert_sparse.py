# sparse - upsert with duplicate indices


import pytest
from pinecone import Vector, SparseValues, PineconeApiException
from .conftest import build_asyncioindex_client, poll_for_freshness
from ..helpers import random_string, embedding_values

@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [
    random_string(20)
])
async def test_upsert_with_batch_size_sparse(pc, sparse_index_host, target_namespace):
    asyncio_sparse_idx = build_asyncioindex_client(pc, sparse_index_host)
    
    await asyncio_sparse_idx.upsert(
        vectors=[
            Vector(
                id=str(i),
                sparse_values=SparseValues(
                    indices=[j for j in range(100)], 
                    values=embedding_values(100)
                ),
            )
            for i in range(100)
        ],
        namespace=target_namespace,
        batch_size=10,
        show_progress=False,
    )
    
    await poll_for_freshness(asyncio_sparse_idx, target_namespace, 100)
    
    # Upsert with invalid batch size
    with pytest.raises(ValueError) as e:
        await asyncio_sparse_idx.upsert(
            vectors=[
                Vector(
                    id=str(i),
                    sparse_values=SparseValues(
                        indices=[j for j in range(100)], 
                        values=embedding_values(100)
                    ),
                )
                for i in range(100)
            ],
            namespace=target_namespace,
            batch_size=0,
        )
    assert "batch_size must be a positive integer" in str(e)
    
    # When upserting with duplicate indices
    with pytest.raises(PineconeApiException) as e:
        await asyncio_sparse_idx.upsert(
            vectors=[
                Vector(
                    id="1",
                    sparse_values=SparseValues(
                        indices=[1, 2, 1], 
                        values=[0.1, 0.2, 0.3]
                    ),
                )
            ],
            namespace=target_namespace,
            batch_size=10,
        )
    