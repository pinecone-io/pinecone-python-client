import pytest
import logging
from pinecone import Vector
from .conftest import build_asyncioindex_client, poll_until_lsn_reconciled_async
from tests.integration.helpers import random_string, embedding_values

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
async def test_delete_by_ids(index_host, dimension, target_namespace):
    """Test deleting vectors by IDs in asyncio"""
    asyncio_idx = build_asyncioindex_client(index_host)

    try:
        # Upsert some vectors
        vectors_to_upsert = [
            Vector(id=f"vec_{i}", values=embedding_values(dimension)) for i in range(5)
        ]
        upsert_response = await asyncio_idx.upsert(
            vectors=vectors_to_upsert, namespace=target_namespace, show_progress=False
        )

        # Wait for upsert to be indexed
        await poll_until_lsn_reconciled_async(
            asyncio_idx, upsert_response._response_info, namespace=target_namespace
        )

        # Verify vectors exist
        fetch_response = await asyncio_idx.fetch(
            ids=["vec_0", "vec_1"], namespace=target_namespace
        )
        assert len(fetch_response.vectors) == 2

        # Delete specific vectors by IDs
        delete_response = await asyncio_idx.delete(
            ids=["vec_0", "vec_1"], namespace=target_namespace
        )
        logger.info(f"Delete response: {delete_response}")

        # Verify deletion - this is the critical part that was failing
        assert delete_response is not None
        assert isinstance(delete_response, dict)

        # Wait for delete to be indexed
        await poll_until_lsn_reconciled_async(
            asyncio_idx, delete_response.get("_response_info", {}), namespace=target_namespace
        )

        # Verify vectors are deleted
        fetch_response = await asyncio_idx.fetch(
            ids=["vec_0", "vec_1"], namespace=target_namespace
        )
        assert len(fetch_response.vectors) == 0

        # Verify remaining vectors still exist
        fetch_response = await asyncio_idx.fetch(
            ids=["vec_2", "vec_3", "vec_4"], namespace=target_namespace
        )
        assert len(fetch_response.vectors) == 3

    finally:
        await asyncio_idx.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
async def test_delete_all_in_namespace(index_host, dimension, target_namespace):
    """Test deleting all vectors in a namespace - the original bug scenario"""
    asyncio_idx = build_asyncioindex_client(index_host)

    try:
        # Upsert some vectors
        vectors_to_upsert = [
            Vector(id=f"vec_{i}", values=embedding_values(dimension)) for i in range(10)
        ]
        upsert_response = await asyncio_idx.upsert(
            vectors=vectors_to_upsert, namespace=target_namespace, show_progress=False
        )

        # Wait for upsert to be indexed
        await poll_until_lsn_reconciled_async(
            asyncio_idx, upsert_response._response_info, namespace=target_namespace
        )

        # Verify vectors exist
        stats = await asyncio_idx.describe_index_stats()
        namespace_stats = stats.namespaces.get(target_namespace)
        assert namespace_stats is not None
        assert namespace_stats.vector_count == 10

        # Delete all vectors in namespace - THIS WAS FAILING WITH AttributeError
        delete_response = await asyncio_idx.delete(
            delete_all=True, namespace=target_namespace
        )
        logger.info(f"Delete all response: {delete_response}")

        # Verify the response doesn't cause AttributeError
        assert delete_response is not None
        assert isinstance(delete_response, dict)

        # Wait for delete to be indexed
        if "_response_info" in delete_response:
            await poll_until_lsn_reconciled_async(
                asyncio_idx, delete_response["_response_info"], namespace=target_namespace
            )

        # Verify all vectors are deleted
        stats = await asyncio_idx.describe_index_stats()
        namespace_stats = stats.namespaces.get(target_namespace)
        # Namespace might not exist anymore or have 0 vectors
        assert namespace_stats is None or namespace_stats.vector_count == 0

    finally:
        await asyncio_idx.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("target_namespace", [random_string(20)])
async def test_delete_by_filter(index_host, dimension, target_namespace):
    """Test deleting vectors by filter in asyncio"""
    asyncio_idx = build_asyncioindex_client(index_host)

    try:
        # Upsert vectors with metadata
        vectors_to_upsert = [
            Vector(
                id=f"vec_{i}",
                values=embedding_values(dimension),
                metadata={"category": "A" if i % 2 == 0 else "B"},
            )
            for i in range(10)
        ]
        upsert_response = await asyncio_idx.upsert(
            vectors=vectors_to_upsert, namespace=target_namespace, show_progress=False
        )

        # Wait for upsert to be indexed
        await poll_until_lsn_reconciled_async(
            asyncio_idx, upsert_response._response_info, namespace=target_namespace
        )

        # Delete vectors with filter
        delete_response = await asyncio_idx.delete(
            filter={"category": {"$eq": "A"}}, namespace=target_namespace
        )
        logger.info(f"Delete by filter response: {delete_response}")

        # Verify deletion response
        assert delete_response is not None
        assert isinstance(delete_response, dict)

        # Wait for delete to be indexed
        if "_response_info" in delete_response:
            await poll_until_lsn_reconciled_async(
                asyncio_idx, delete_response["_response_info"], namespace=target_namespace
            )

        # Verify only category A vectors are deleted (approximately 5 vectors)
        stats = await asyncio_idx.describe_index_stats()
        namespace_stats = stats.namespaces.get(target_namespace)
        # Should have about 5 vectors remaining (category B)
        assert namespace_stats is not None
        assert namespace_stats.vector_count <= 5

    finally:
        await asyncio_idx.close()


@pytest.mark.asyncio
async def test_delete_response_has_response_info(index_host, dimension):
    """Test that delete response includes _response_info metadata"""
    asyncio_idx = build_asyncioindex_client(index_host)
    target_namespace = random_string(20)

    try:
        # Upsert a vector
        upsert_response = await asyncio_idx.upsert(
            vectors=[Vector(id="test_vec", values=embedding_values(dimension))],
            namespace=target_namespace,
            show_progress=False,
        )

        # Wait for upsert to be indexed
        await poll_until_lsn_reconciled_async(
            asyncio_idx, upsert_response._response_info, namespace=target_namespace
        )

        # Delete the vector
        delete_response = await asyncio_idx.delete(ids=["test_vec"], namespace=target_namespace)

        # Verify response structure - this validates the fix
        assert isinstance(delete_response, dict)
        # _response_info should be present for dict responses
        assert "_response_info" in delete_response
        assert "raw_headers" in delete_response["_response_info"]

        logger.info(f"Delete response with metadata: {delete_response}")

    finally:
        await asyncio_idx.close()
