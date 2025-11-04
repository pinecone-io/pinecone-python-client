import time
import logging

from pinecone import NamespaceDescription

logger = logging.getLogger(__name__)


def setup_namespace_data(index, namespace: str, num_vectors: int = 2):
    """Helper function to set up test data in a namespace"""
    vectors = [(f"id_{i}", [0.1, 0.2]) for i in range(num_vectors)]
    index.upsert(vectors=vectors, namespace=namespace)
    # Wait for data to be upserted
    time.sleep(5)


def verify_namespace_exists(index, namespace: str) -> bool:
    """Helper function to verify if a namespace exists"""
    try:
        index.describe_namespace(namespace=namespace)
        return True
    except Exception:
        return False


def delete_all_namespaces(index):
    """Helper function to delete all namespaces in an index"""
    try:
        # Get all namespaces
        namespaces = list(index.list_namespaces())

        # Delete each namespace
        for namespace in namespaces:
            try:
                index.delete_namespace(namespace=namespace.name)
            except Exception as e:
                logger.error(f"Error deleting namespace {namespace.name}: {e}")

        # Wait for deletions to complete
        time.sleep(5)
    except Exception as e:
        logger.error(f"Error in delete_all_namespaces: {e}")


class TestNamespaceOperations:
    def test_create_namespace(self, idx):
        """Test creating a namespace"""
        test_namespace = "test_create_namespace_sync"

        try:
            # Ensure namespace doesn't exist first
            if verify_namespace_exists(idx, test_namespace):
                idx.delete_namespace(namespace=test_namespace)
                time.sleep(10)

            # Create namespace
            description = idx.create_namespace(name=test_namespace)

            # Verify namespace was created
            assert isinstance(description, NamespaceDescription)
            assert description.name == test_namespace
            # New namespace should have 0 records (record_count may be None, 0, or "0" as string)
            assert (
                description.record_count is None
                or description.record_count == 0
                or description.record_count == "0"
            )

            # Verify namespace exists by describing it
            # Namespace may not be immediately available after creation, so retry with backoff
            max_retries = 5
            retry_delay = 2
            for attempt in range(max_retries):
                try:
                    verify_description = idx.describe_namespace(namespace=test_namespace)
                    assert verify_description.name == test_namespace
                    break
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(retry_delay)

        finally:
            # Cleanup
            if verify_namespace_exists(idx, test_namespace):
                idx.delete_namespace(namespace=test_namespace)
                time.sleep(10)

    def test_create_namespace_duplicate(self, idx):
        """Test creating a duplicate namespace raises an error"""
        test_namespace = "test_create_duplicate_sync"

        try:
            # Ensure namespace doesn't exist first
            if verify_namespace_exists(idx, test_namespace):
                idx.delete_namespace(namespace=test_namespace)
                time.sleep(10)

            # Create namespace first time
            description = idx.create_namespace(name=test_namespace)
            assert description.name == test_namespace

            # Try to create duplicate namespace - should raise an error
            # GRPC errors raise PineconeException, not PineconeApiException
            import pytest
            from pinecone.exceptions import PineconeException

            with pytest.raises(PineconeException):
                idx.create_namespace(name=test_namespace)

        finally:
            # Cleanup
            if verify_namespace_exists(idx, test_namespace):
                idx.delete_namespace(namespace=test_namespace)
                time.sleep(10)

    def test_describe_namespace(self, idx):
        """Test describing a namespace"""
        # Setup test data
        test_namespace = "test_describe_namespace_sync"
        setup_namespace_data(idx, test_namespace)

        try:
            # Test describe
            description = idx.describe_namespace(namespace=test_namespace)
            assert isinstance(description, NamespaceDescription)
            assert description.name == test_namespace
        finally:
            # Delete all namespaces before next test is run
            delete_all_namespaces(idx)

    def test_delete_namespace(self, idx):
        """Test deleting a namespace"""
        # Setup test data
        test_namespace = "test_delete_namespace_sync"
        setup_namespace_data(idx, test_namespace)

        # Verify namespace exists
        assert verify_namespace_exists(idx, test_namespace)

        # Delete namespace
        idx.delete_namespace(namespace=test_namespace)

        # Wait for namespace to be deleted
        time.sleep(10)

        # Verify namespace is deleted
        assert not verify_namespace_exists(idx, test_namespace)

    def test_list_namespaces(self, idx):
        """Test listing namespaces"""
        # Create multiple test namespaces
        test_namespaces = ["test_list_1", "test_list_2", "test_list_3"]
        for ns in test_namespaces:
            setup_namespace_data(idx, ns)

        try:
            # Get all namespaces
            namespaces = list(idx.list_namespaces())

            # Verify results
            assert len(namespaces) == len(test_namespaces)
            namespace_names = [ns.name for ns in namespaces]
            for test_ns in test_namespaces:
                assert test_ns in namespace_names

            # Verify each namespace has correct structure
            for ns in namespaces:
                assert isinstance(ns, NamespaceDescription)
                assert hasattr(ns, "name")
                assert hasattr(ns, "vector_count")
        finally:
            # Delete all namespaces before next test is run
            delete_all_namespaces(idx)

    def test_list_namespaces_with_limit(self, idx):
        """Test listing namespaces with limit"""
        # Create multiple test namespaces
        test_namespaces = [f"test_limit_{i}" for i in range(5)]
        for ns in test_namespaces:
            setup_namespace_data(idx, ns)

        try:
            # Get namespaces with limit
            namespaces = list(idx.list_namespaces(limit=2))

            # Verify results
            assert len(namespaces) >= 2  # Should get at least 2 namespaces
            for ns in namespaces:
                assert isinstance(ns, NamespaceDescription)
                assert hasattr(ns, "name")
                assert hasattr(ns, "vector_count")

        finally:
            # Delete all namespaces before next test is run
            delete_all_namespaces(idx)

    def test_list_namespaces_paginated(self, idx):
        """Test listing namespaces with pagination"""
        # Create multiple test namespaces
        test_namespaces = [f"test_paginated_{i}" for i in range(5)]
        for ns in test_namespaces:
            setup_namespace_data(idx, ns)

        try:
            # Get first page
            response = idx.list_namespaces_paginated(limit=2)
            assert len(response.namespaces) == 2
            assert response.pagination.next is not None

            # Get second page
            next_response = idx.list_namespaces_paginated(
                limit=2, pagination_token=response.pagination.next
            )
            assert len(next_response.namespaces) == 2
            assert next_response.pagination.next is not None

            # Get final page
            final_response = idx.list_namespaces_paginated(
                limit=2, pagination_token=next_response.pagination.next
            )
            assert len(final_response.namespaces) == 1
            assert final_response.pagination is None
        finally:
            delete_all_namespaces(idx)
