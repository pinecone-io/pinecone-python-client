import pytest
import asyncio
from typing import List

from pinecone.core.openapi.db_data.models import NamespaceDescription
from tests.integration.data_asyncio.conftest import build_asyncioindex_client


async def setup_namespace_data(index, namespace: str, num_vectors: int = 3):
    """Helper function to set up test data in a namespace"""
    vectors = [(f"id_{i}", [0.1, 0.2]) for i in range(num_vectors)]
    await index.upsert(vectors=vectors, namespace=namespace)
    # Wait for vectors to be upserted
    await asyncio.sleep(5)


async def verify_namespace_exists(index, namespace: str) -> bool:
    """Helper function to verify if a namespace exists"""
    try:
        await index.describe_namespace(namespace)
        return True
    except Exception:
        return False


# async def get_namespace_names(index) -> List[str]:
#     """Helper function to get all namespace names"""
#     return [ns.name async for ns in index.list_namespaces()]


class TestNamespaceOperationsAsyncio:
    @pytest.mark.asyncio
    async def test_describe_namespace(self, index_host):
        """Test describing a namespace"""
        asyncio_idx = build_asyncioindex_client(index_host)

        # Setup test data
        test_namespace = "test_describe_namespace_async"
        await setup_namespace_data(asyncio_idx, test_namespace)

        # Test describe
        description = await asyncio_idx.describe_namespace(test_namespace)
        assert isinstance(description, NamespaceDescription)
        assert description.name == test_namespace

    @pytest.mark.asyncio
    async def test_delete_namespace(self, index_host):
        """Test deleting a namespace"""
        asyncio_idx = build_asyncioindex_client(index_host)
        # Setup test data
        test_namespace = "test_delete_namespace_async"
        await setup_namespace_data(asyncio_idx, test_namespace)

        # Verify namespace exists
        assert await verify_namespace_exists(asyncio_idx, test_namespace)

        # Delete namespace
        await asyncio_idx.delete_namespace(test_namespace)

        # Wait for namespace to be deleted
        await asyncio.sleep(5)

        # Verify namespace is deleted
        assert not await verify_namespace_exists(asyncio_idx, test_namespace)

    # @pytest.mark.asyncio
    # async def test_list_namespaces(self, index_asyncio):
    #     """Test listing namespaces"""
    #     # Create multiple test namespaces
    #     test_namespaces = ["test_list_1_async", "test_list_2_async", "test_list_3_async"]
    #     for ns in test_namespaces:
    #         await setup_namespace_data(index_asyncio, ns)
    #
    #     # Get all namespaces
    #     namespaces = [ns async for ns in index_asyncio.list_namespaces()]
    #
    #     # Verify results
    #     assert len(namespaces) >= len(test_namespaces)
    #     namespace_names = [ns.name for ns in namespaces]
    #     for test_ns in test_namespaces:
    #         assert test_ns in namespace_names
    #
    #     # Verify each namespace has correct structure
    #     for ns in namespaces:
    #         assert isinstance(ns, NamespaceDescription)
    #         assert hasattr(ns, 'name')
    #         assert hasattr(ns, 'vector_count')
    #
    # @pytest.mark.asyncio
    # async def test_namespace_operations_with_pagination(self, index_asyncio):
    #     """Test namespace operations with pagination"""
    #     # Create many namespaces to test pagination
    #     test_namespaces = [f"test_pagination_async_{i}" for i in range(15)]  # More than default page size
    #     for ns in test_namespaces:
    #         await setup_namespace_data(index_asyncio, ns)
    #
    #     # Test listing with limit
    #     namespaces = [ns async for ns in index_asyncio.list_namespaces(limit=5)]
    #     assert len(namespaces) >= 5  # Should get at least 5 namespaces
    #
    #     # Test listing all namespaces
    #     all_namespaces = [ns async for ns in index_asyncio.list_namespaces()]
    #     assert len(all_namespaces) >= len(test_namespaces)
    #     namespace_names = [ns.name for ns in all_namespaces]
    #     for test_ns in test_namespaces:
    #         assert test_ns in namespace_names
    #
    # @pytest.mark.asyncio
    # async def test_namespace_operations_with_invalid_namespace(self, index_asyncio):
    #     """Test namespace operations with invalid namespace"""
    #     invalid_namespace = "non_existent_namespace_async"
    #
    #     # Test describe with invalid namespace
    #     with pytest.raises(Exception):
    #         await index_asyncio.describe_namespace(invalid_namespace)
    #
    #     # Test delete with invalid namespace
    #     with pytest.raises(Exception):
    #         await index_asyncio.delete_namespace(invalid_namespace)

    # @pytest.mark.asyncio
    # async def test_namespace_operations_with_empty_namespace(self, index_asyncio):
    #     """Test namespace operations with empty namespace"""
    #     empty_namespace = "test_empty_namespace_async"
    #
    #     # Create empty namespace
    #     await index_asyncio.upsert(vectors=[], namespace=empty_namespace)
    #     await asyncio.sleep(5)
    #
    #     # Test describe
    #     description = await index_asyncio.describe_namespace(empty_namespace)
    #     assert description.name == empty_namespace
    #     assert description.vector_count == 0
    #
    #     # Test list includes empty namespace
    #     namespaces = [ns async for ns in index_asyncio.list_namespaces()]
    #     namespace_names = [ns.name for ns in namespaces]
    #     assert empty_namespace in namespace_names
    #
    # @pytest.mark.asyncio
    # async def test_concurrent_namespace_operations(self, index_asyncio):
    #     """Test concurrent namespace operations"""
    #     # Create multiple namespaces concurrently
    #     test_namespaces = [f"test_concurrent_{i}" for i in range(5)]
    #     setup_tasks = [setup_namespace_data(index_asyncio, ns) for ns in test_namespaces]
    #     await asyncio.gather(*setup_tasks)
    #
    #     # Perform multiple describe operations concurrently
    #     describe_tasks = [index_asyncio.describe_namespace(ns) for ns in test_namespaces]
    #     descriptions = await asyncio.gather(*describe_tasks)
    #
    #     # Verify all descriptions
    #     for ns, desc in zip(test_namespaces, descriptions):
    #         assert desc.name == ns
    #         assert desc.vector_count >= 10
    #
    #     # Delete all namespaces concurrently
    #     delete_tasks = [index_asyncio.delete_namespace(ns) for ns in test_namespaces]
    #     await asyncio.gather(*delete_tasks)
    #
    #     # Verify all namespaces are deleted
    #     for ns in test_namespaces:
    #         assert not await verify_namespace_exists(index_asyncio, ns)