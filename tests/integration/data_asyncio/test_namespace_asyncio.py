import pytest
import asyncio

from pinecone.core.openapi.db_data.models import NamespaceDescription
from tests.integration.data_asyncio.conftest import build_asyncioindex_client


async def setup_namespace_data(index, namespace: str, num_vectors: int = 2):
    """Helper function to set up test data in a namespace"""
    vectors = [(f"id_{i}", [0.1, 0.2]) for i in range(num_vectors)]
    await index.upsert(vectors=vectors, namespace=namespace)
    # Wait for vectors to be upserted
    await asyncio.sleep(5)


async def verify_namespace_exists(index, namespace: str) -> bool:
    """Helper function to verify if a namespace exists"""
    try:
        await index.namespace.describe(namespace)
        return True
    except Exception:
        return False


async def delete_all_namespaces(index):
    """Helper function to delete all namespaces in an index"""
    try:
        # Get all namespaces
        namespaces = await index.namespace.list_paginated()

        # Delete each namespace
        for namespace in namespaces.namespaces:
            try:
                await index.namespace.delete(namespace.name)
            except Exception as e:
                print(f"Error deleting namespace {namespace.name}: {e}")

        # Wait for deletions to complete
        await asyncio.sleep(5)
    except Exception as e:
        print(f"Error in delete_all_namespaces: {e}")


class TestNamespaceOperationsAsyncio:
    @pytest.mark.asyncio
    async def test_describe_namespace(self, index_host):
        """Test describing a namespace"""
        asyncio_idx = build_asyncioindex_client(index_host)

        # Setup test data
        test_namespace = "test_describe_namespace_async"
        await setup_namespace_data(asyncio_idx, test_namespace)

        try:
            # Test describe
            description = await asyncio_idx.namespace.describe(test_namespace)
            assert isinstance(description, NamespaceDescription)
            assert description.name == test_namespace
        finally:
            # Delete all namespaces before next test is run
            await delete_all_namespaces(asyncio_idx)

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
        await asyncio_idx.namespace.delete(test_namespace)

        # Wait for namespace to be deleted
        await asyncio.sleep(10)

        # Verify namespace is deleted
        assert not await verify_namespace_exists(asyncio_idx, test_namespace)

    @pytest.mark.asyncio
    async def test_list_namespaces(self, index_host):
        """Test listing namespaces"""
        asyncio_idx = build_asyncioindex_client(index_host)
        # Create multiple test namespaces
        test_namespaces = ["test_list_1_async", "test_list_2_async", "test_list_3_async"]
        for ns in test_namespaces:
            await setup_namespace_data(asyncio_idx, ns)

        try:
            # Get all namespaces
            namespaces = await asyncio_idx.namespace.list_paginated()

            # Verify results
            assert len(namespaces.namespaces) >= len(test_namespaces)
            namespace_names = [ns.name for ns in namespaces.namespaces]
            for test_ns in test_namespaces:
                assert test_ns in namespace_names

            # Verify each namespace has correct structure
            for ns in namespaces.namespaces:
                assert isinstance(ns, NamespaceDescription)
                assert hasattr(ns, "name")
                assert hasattr(ns, "vector_count")
        finally:
            # Delete all namespaces before next test is run
            await delete_all_namespaces(asyncio_idx)

    @pytest.mark.asyncio
    async def test_list_namespaces_with_limit(self, index_host):
        """Test listing namespaces with limit"""
        asyncio_idx = build_asyncioindex_client(index_host)
        # Create multiple test namespaces
        test_namespaces = [f"test_limit_async_{i}" for i in range(5)]
        for ns in test_namespaces:
            await setup_namespace_data(asyncio_idx, ns)

        try:
            # Get namespaces with limit
            namespaces = await asyncio_idx.namespace.list_paginated(limit=2)

            # Verify results
            assert len(namespaces.namespaces) == 2  # Should get exactly 2 namespaces
            for ns in namespaces.namespaces:
                assert isinstance(ns, NamespaceDescription)
                assert hasattr(ns, "name")
                assert hasattr(ns, "vector_count")
        finally:
            # Delete all namespaces before next test is run
            await delete_all_namespaces(asyncio_idx)

    @pytest.mark.asyncio
    async def test_list_namespaces_paginated(self, index_host):
        """Test listing namespaces with pagination"""
        asyncio_idx = build_asyncioindex_client(index_host)
        # Create multiple test namespaces
        test_namespaces = [f"test_paginated_async_{i}" for i in range(5)]
        for ns in test_namespaces:
            await setup_namespace_data(asyncio_idx, ns)

        try:
            # Get first page
            response = await asyncio_idx.namespace.list_paginated(limit=2)
            assert len(response.namespaces) == 2
            assert response.pagination.next is not None

            # Get second page
            next_response = await asyncio_idx.namespace.list_paginated(
                limit=2, pagination_token=response.pagination.next
            )
            assert len(next_response.namespaces) == 2
            assert next_response.pagination.next is not None

            # Get final page
            final_response = await asyncio_idx.namespace.list_paginated(
                limit=2, pagination_token=next_response.pagination.next
            )
            assert len(final_response.namespaces) == 1
            assert final_response.pagination is None
        finally:
            # Delete all namespaces before next test is run
            await delete_all_namespaces(asyncio_idx)
