import pytest
import logging

from pinecone import NamespaceDescription
from .conftest import build_asyncioindex_client, poll_until_lsn_reconciled_async
from tests.integration.helpers import random_string

logger = logging.getLogger(__name__)


async def setup_namespace_data(index, namespace: str, num_vectors: int = 2):
    """Helper function to set up test data in a namespace"""
    vectors = [(f"id_{i}", [0.1, 0.2]) for i in range(num_vectors)]
    upsert1 = await index.upsert(vectors=vectors, namespace=namespace)
    await poll_until_lsn_reconciled_async(index, upsert1._response_info, namespace=namespace)


async def verify_namespace_exists(index, namespace: str) -> bool:
    """Helper function to verify if a namespace exists"""
    try:
        description = await index.describe_namespace(namespace=namespace)
        logger.info(f"Verified namespace {namespace} with description: {description}")
        assert description.name == namespace
        return True
    except Exception:
        return False


async def delete_all_namespaces(index):
    """Helper function to delete all namespaces in an index"""
    try:
        # Get all namespaces
        namespaces = await index.list_namespaces_paginated()

        # Delete each namespace
        for namespace in namespaces.namespaces:
            try:
                resp = await index.delete_namespace(namespace=namespace.name)
                logger.info(f"Deleted namespace {namespace.name} with response: {resp}")
            except Exception as e:
                logger.error(f"Error deleting namespace {namespace.name}: {e}")
    except Exception as e:
        logger.error(f"Error in delete_all_namespaces: {e}")


class TestNamespaceOperationsAsyncio:
    @pytest.mark.asyncio
    async def test_create_namespace(self, index_host):
        """Test creating a namespace"""
        asyncio_idx = build_asyncioindex_client(index_host)
        test_namespace = random_string(10)

        try:
            # Create namespace
            ns_description = await asyncio_idx.create_namespace(name=test_namespace)
            logger.info(f"Created namespace {test_namespace} with description: {ns_description}")

            # Verify namespace was created
            assert isinstance(ns_description, NamespaceDescription)
            assert ns_description.name == test_namespace
            # New namespace should have 0 records (record_count may be None, 0, or "0" as string)
            assert (
                ns_description.record_count is None
                or ns_description.record_count == 0
                or ns_description.record_count == "0"
            )

            # Verify namespace exists by describing it
            verify_description = await asyncio_idx.describe_namespace(namespace=test_namespace)
            assert verify_description.name == test_namespace

        finally:
            if await verify_namespace_exists(asyncio_idx, test_namespace):
                await asyncio_idx.delete_namespace(namespace=test_namespace)
            await asyncio_idx.close()

    @pytest.mark.asyncio
    async def test_create_namespace_duplicate(self, index_host):
        """Test creating a duplicate namespace raises an error"""
        asyncio_idx = build_asyncioindex_client(index_host)
        test_namespace = random_string(10)

        try:
            # Create namespace first time
            ns_description = await asyncio_idx.create_namespace(name=test_namespace)
            assert ns_description.name == test_namespace

            # Try to create duplicate namespace - should raise an error
            from pinecone.exceptions import PineconeApiException

            with pytest.raises(PineconeApiException):
                await asyncio_idx.create_namespace(name=test_namespace)

        finally:
            # Cleanup
            if await verify_namespace_exists(asyncio_idx, test_namespace):
                await asyncio_idx.delete_namespace(namespace=test_namespace)
            await asyncio_idx.close()

    @pytest.mark.asyncio
    async def test_describe_namespace(self, index_host):
        """Test describing a namespace"""
        asyncio_idx = build_asyncioindex_client(index_host)

        # Setup test data
        test_namespace = random_string(10)
        await setup_namespace_data(asyncio_idx, test_namespace)

        try:
            # Test describe
            ns_description = await asyncio_idx.describe_namespace(namespace=test_namespace)
            assert isinstance(ns_description, NamespaceDescription)
            assert ns_description.name == test_namespace
        finally:
            # Delete all namespaces before next test is run
            await delete_all_namespaces(asyncio_idx)
            await asyncio_idx.close()

    @pytest.mark.asyncio
    async def test_delete_namespace(self, index_host):
        """Test deleting a namespace"""
        try:
            asyncio_idx = build_asyncioindex_client(index_host)
            # Setup test data
            test_namespace = random_string(10)
            await setup_namespace_data(asyncio_idx, test_namespace)

            # Verify namespace exists
            assert await verify_namespace_exists(asyncio_idx, test_namespace)

            # Delete namespace
            resp = await asyncio_idx.delete_namespace(namespace=test_namespace)
            logger.info(f"Deleted namespace {test_namespace} with response: {resp}")

        finally:
            await asyncio_idx.close()

    @pytest.mark.asyncio
    async def test_list_namespaces(self, index_host):
        """Test listing namespaces"""
        asyncio_idx = build_asyncioindex_client(index_host)
        # Create multiple test namespaces
        test_namespaces = [random_string(20) for _ in range(3)]
        for ns in test_namespaces:
            await setup_namespace_data(asyncio_idx, ns)

        try:
            # Get all namespaces
            async for ns in asyncio_idx.list_namespaces():
                assert isinstance(ns, NamespaceDescription)
                assert ns.name in test_namespaces
                assert int(ns.record_count) == 2

        finally:
            await delete_all_namespaces(asyncio_idx)
            await asyncio_idx.close()

    @pytest.mark.asyncio
    async def test_list_namespaces_with_limit(self, index_host):
        """Test listing namespaces with limit"""
        asyncio_idx = build_asyncioindex_client(index_host)
        # Create multiple test namespaces
        test_namespaces = [random_string(20) for i in range(5)]
        for ns in test_namespaces:
            await setup_namespace_data(asyncio_idx, ns)

        try:
            # Get namespaces with limit
            namespaces = await asyncio_idx.list_namespaces_paginated(limit=2)

            # First page
            assert len(namespaces.namespaces) == 2  # Should get exactly 2 namespaces
            for ns in namespaces.namespaces:
                assert isinstance(ns, NamespaceDescription)
                assert ns.name is not None
                assert ns.record_count is not None
            assert namespaces.pagination.next is not None

            listed_namespaces = []
            async for ns in asyncio_idx.list_namespaces():
                listed_namespaces.append(ns.name)
            for test_ns in test_namespaces:
                assert test_ns in listed_namespaces
        finally:
            # Delete all namespaces before next test is run
            await delete_all_namespaces(asyncio_idx)
            await asyncio_idx.close()
