import os
import time

import pytest

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
        index.namespace.describe(namespace)
        return True
    except Exception:
        return False


def delete_all_namespaces(index):
    """Helper function to delete all namespaces in an index"""
    try:
        # Get all namespaces
        namespaces = list(index.namespace.list())

        # Delete each namespace
        for namespace in namespaces:
            try:
                index.namespace.delete(namespace.name)
            except Exception as e:
                print(f"Error deleting namespace {namespace.name}: {e}")

        # Wait for deletions to complete
        time.sleep(5)
    except Exception as e:
        print(f"Error in delete_all_namespaces: {e}")


@pytest.mark.skipif(
    os.getenv("USE_GRPC") == "true", reason="Disable until grpc namespaces support is added"
)
class TestNamespaceOperations:
    def test_describe_namespace(self, idx):
        """Test describing a namespace"""
        # Setup test data
        test_namespace = "test_describe_namespace_sync"
        setup_namespace_data(idx, test_namespace)

        try:
            # Test describe
            description = idx.namespace.describe(test_namespace)
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
        idx.namespace.delete(test_namespace)

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
            namespaces = list(idx.namespace.list())

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
            namespaces = list(idx.namespace.list(limit=2))

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
            response = idx.namespace.list_paginated(limit=2)
            assert len(response.namespaces) == 2
            assert response.pagination.next is not None

            # Get second page
            next_response = idx.namespace.list_paginated(
                limit=2, pagination_token=response.pagination.next
            )
            assert len(next_response.namespaces) == 2
            assert next_response.pagination.next is not None

            # Get final page
            final_response = idx.namespace.list_paginated(
                limit=2, pagination_token=next_response.pagination.next
            )
            assert len(final_response.namespaces) == 1
            assert final_response.pagination is None
        finally:
            delete_all_namespaces(idx)
