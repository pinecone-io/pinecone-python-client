# sparse - upsert with duplicate indices


import pytest
from pinecone import Vector, SparseValues, PineconeApiException
from .conftest import build_asyncioindex_client, poll_until_lsn_reconciled_async
from tests.integration.helpers import random_string, embedding_values

import logging

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
async def test_upsert_with_batch_size_sparse(sparse_index_host, target_namespace):
    asyncio_sparse_idx = build_asyncioindex_client(sparse_index_host)

    upsert1 = await asyncio_sparse_idx.upsert(
        vectors=[
            Vector(
                id=str(i),
                sparse_values=SparseValues(
                    indices=[j for j in range(100)], values=embedding_values(100)
                ),
            )
            for i in range(100)
        ],
        namespace=target_namespace,
        batch_size=10,
        show_progress=False,
    )

    await poll_until_lsn_reconciled_async(
        asyncio_sparse_idx, upsert1._response_info, namespace=target_namespace
    )

    # Upsert with invalid batch size
    with pytest.raises(ValueError) as e:
        await asyncio_sparse_idx.upsert(
            vectors=[
                Vector(
                    id=str(i),
                    sparse_values=SparseValues(
                        indices=[j for j in range(100)], values=embedding_values(100)
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
                    id="1", sparse_values=SparseValues(indices=[1, 2, 1], values=[0.1, 0.2, 0.3])
                )
            ],
            namespace=target_namespace,
            batch_size=10,
        )

    await poll_until_lsn_reconciled_async(
        asyncio_sparse_idx, upsert1._response_info, namespace=target_namespace
    )

    fetched_vec = await asyncio_sparse_idx.fetch(ids=["1", "2", "3"], namespace=target_namespace)
    assert len(fetched_vec.vectors.keys()) == 3
    assert "1" in fetched_vec.vectors
    assert "2" in fetched_vec.vectors
    assert "3" in fetched_vec.vectors

    assert fetched_vec._response_info is not None, (
        "Expected _response_info to be present on fetch response"
    )
    logger.info(f"Fetch response info: {fetched_vec._response_info}")
    await asyncio_sparse_idx.close()
