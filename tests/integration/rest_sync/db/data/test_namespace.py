import pytest
import logging
from tests.integration.helpers import poll_until_lsn_reconciled, random_string

from pinecone import NamespaceDescription

logger = logging.getLogger(__name__)


def setup_namespace_data(index, namespace: str, num_vectors: int = 2):
    """Helper function to set up test data in a namespace"""
    vectors = [(f"id_{i}", [0.1, 0.2]) for i in range(num_vectors)]
    upsert1 = index.upsert(vectors=vectors, namespace=namespace)
    poll_until_lsn_reconciled(index, upsert1._response_info, namespace=namespace)


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
    except Exception as e:
        logger.error(f"Error in delete_all_namespaces: {e}")


class TestNamespaceOperations:
    def test_create_namespace(self, idx):
        """Test creating a namespace"""
        test_namespace = random_string(10)

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
        verify_description = idx.describe_namespace(namespace=test_namespace)
        assert verify_description.name == test_namespace

    def test_create_namespace_duplicate(self, idx):
        """Test creating a duplicate namespace raises an error"""
        test_namespace = random_string(10)

        # Create namespace first time
        description = idx.create_namespace(name=test_namespace)
        assert description.name == test_namespace

        # Try to create duplicate namespace - should raise an error
        # GRPC errors raise PineconeException, not PineconeApiException
        from pinecone.exceptions import PineconeException

        with pytest.raises(PineconeException):
            idx.create_namespace(name=test_namespace)

    def test_describe_namespace(self, idx):
        """Test describing a namespace"""
        # Setup test data
        test_namespace = random_string(10)
        setup_namespace_data(idx, test_namespace)

        # Test describe
        description = idx.describe_namespace(namespace=test_namespace)
        assert isinstance(description, NamespaceDescription)
        assert description.name == test_namespace
        assert description._response_info is not None

        idx.delete_namespace(namespace=test_namespace)

    def test_delete_namespace(self, idx):
        """Test deleting a namespace"""
        # Setup test data
        test_namespace = random_string(10)
        setup_namespace_data(idx, test_namespace)

        # Verify namespace exists
        assert verify_namespace_exists(idx, test_namespace)

        # Delete namespace
        idx.delete_namespace(namespace=test_namespace)

    def test_list_namespaces(self, idx):
        """Test listing namespaces"""
        # Create multiple test namespaces
        test_namespaces = [random_string(10) for _ in range(3)]
        for ns in test_namespaces:
            setup_namespace_data(idx, ns)

        # Get all namespaces
        namespaces = list(idx.list_namespaces())

        # Verify results
        assert len(namespaces) >= len(test_namespaces)
        namespace_names = [ns.name for ns in namespaces]
        for test_ns in test_namespaces:
            assert test_ns in namespace_names

        # Verify each namespace has correct structure
        for ns in namespaces:
            assert isinstance(ns, NamespaceDescription)
            assert ns.name is not None
            assert ns.record_count is not None
            idx.delete_namespace(namespace=ns.name)

    def test_list_namespaces_with_limit(self, idx):
        """Test listing namespaces with limit"""
        # Create multiple test namespaces
        test_namespaces = [random_string(10) for i in range(5)]
        for ns in test_namespaces:
            setup_namespace_data(idx, ns)

        # Get namespaces with limit
        namespaces = list(idx.list_namespaces(limit=2))

        # Verify results
        assert len(namespaces) >= 2  # Should get at least 2 namespaces
        for ns in namespaces:
            assert isinstance(ns, NamespaceDescription)
            assert hasattr(ns, "name")
            assert hasattr(ns, "record_count")
            idx.delete_namespace(namespace=ns.name)

    def test_list_namespaces_paginated(self, idx):
        """Test listing namespaces with pagination"""
        # Create multiple test namespaces
        test_namespaces = [random_string(10) for i in range(5)]
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
