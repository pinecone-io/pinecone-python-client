import pytest
import time
from typing import List

from pinecone import Pinecone
from pinecone.core.openapi.db_data.models import NamespaceDescription


def setup_namespace_data(index, namespace: str, num_vectors: int = 2):
    """Helper function to set up test data in a namespace"""
    vectors = [(f"id_{i}", [0.1, 0.2]) for i in range(num_vectors)]
    index.upsert(vectors=vectors, namespace=namespace)
    # Wait for data to be upserted
    time.sleep(5)


def verify_namespace_exists(index, namespace: str) -> bool:
    """Helper function to verify if a namespace exists"""
    try:
        index.describe_namespace(namespace)
        return True
    except Exception:
        return False

#
# def get_namespace_names(index) -> List[str]:
#     """Helper function to get all namespace names"""
#     return [ns.name for ns in index.list_namespaces()]


class TestNamespaceOperations:
    def test_describe_namespace(self, idx):
        """Test describing a namespace"""
        # Setup test data
        test_namespace = "test_describe_namespace_sync"
        setup_namespace_data(idx, test_namespace)

        # Test describe
        description = idx.describe_namespace(test_namespace)
        assert isinstance(description, NamespaceDescription)
        assert description.name == test_namespace

    def test_delete_namespace(self, idx):
        """Test deleting a namespace"""
        # Setup test data
        test_namespace = "test_delete_namespace_sync"
        setup_namespace_data(idx, test_namespace)

        # Verify namespace exists
        assert verify_namespace_exists(idx, test_namespace)

        # Delete namespace
        idx.delete_namespace(test_namespace)

        # Wait for namespace to be deleted
        time.sleep(5)

        # Verify namespace is deleted
        assert not verify_namespace_exists(idx, test_namespace)

    # def test_list_namespaces(self, index):
    #     """Test listing namespaces"""
    #     # Create multiple test namespaces
    #     test_namespaces = ["test_list_1", "test_list_2", "test_list_3"]
    #     for ns in test_namespaces:
    #         setup_namespace_data(index, ns)
    #
    #     # Get all namespaces
    #     namespaces = list(index.list_namespaces())
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

    # def test_namespace_operations_with_pagination(self, index):
    #     """Test namespace operations with pagination"""
    #     # Create many namespaces to test pagination
    #     test_namespaces = [f"test_pagination_{i}" for i in range(15)]  # More than default page size
    #     for ns in test_namespaces:
    #         setup_namespace_data(index, ns)
    #
    #     # Test listing with limit
    #     namespaces = list(index.list_namespaces(limit=5))
    #     assert len(namespaces) >= 5  # Should get at least 5 namespaces
    #
    #     # Test listing all namespaces
    #     all_namespaces = list(index.list_namespaces())
    #     assert len(all_namespaces) >= len(test_namespaces)
    #     namespace_names = [ns.name for ns in all_namespaces]
    #     for test_ns in test_namespaces:
    #         assert test_ns in namespace_names
    #
    # def test_namespace_operations_with_invalid_namespace(self, index):
    #     """Test namespace operations with invalid namespace"""
    #     invalid_namespace = "non_existent_namespace"
    #
    #     # Test describe with invalid namespace
    #     with pytest.raises(Exception):
    #         index.describe_namespace(invalid_namespace)
    #
    #     # Test delete with invalid namespace
    #     with pytest.raises(Exception):
    #         index.delete_namespace(invalid_namespace)
    #
    # def test_namespace_operations_with_empty_namespace(self, index):
    #     """Test namespace operations with empty namespace"""
    #     empty_namespace = "test_empty_namespace"
    #
    #     # Create empty namespace
    #     index.upsert(vectors=[], namespace=empty_namespace)
    #     time.sleep(5)
    #
    #     # Test describe
    #     description = index.describe_namespace(empty_namespace)
    #     assert description.name == empty_namespace
    #     assert description.vector_count == 0
    #
    #     # Test list includes empty namespace
    #     namespaces = list(index.list_namespaces())
    #     namespace_names = [ns.name for ns in namespaces]
    #     assert empty_namespace in namespace_names