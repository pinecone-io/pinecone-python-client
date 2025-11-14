import pytest
import time
from pinecone import NamespaceDescription
from tests.integration.helpers import random_string


def verify_namespace_exists(idx, namespace: str) -> bool:
    """Helper function to verify if a namespace exists"""
    try:
        idx.describe_namespace(namespace=namespace)
        return True
    except Exception:
        return False


class TestCreateNamespaceFuture:
    def test_create_namespace_future(self, idx):
        """Test creating a namespace with async_req=True"""
        test_namespace = random_string(20)

        try:
            # Create namespace asynchronously
            future = idx.create_namespace(name=test_namespace, async_req=True)

            # Verify it's a future
            from pinecone.grpc import PineconeGrpcFuture

            assert isinstance(future, PineconeGrpcFuture)

            # Get the result
            description = future.result(timeout=30)

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
            verify_description = idx.describe_namespace(namespace=test_namespace)
            assert verify_description.name == test_namespace

        finally:
            # Cleanup
            if verify_namespace_exists(idx, test_namespace):
                idx.delete_namespace(namespace=test_namespace)
                time.sleep(10)

    def test_create_namespace_future_duplicate(self, idx):
        """Test creating a duplicate namespace raises an error with async_req=True"""
        test_namespace = random_string(20)

        try:
            # Create namespace first time
            future1 = idx.create_namespace(name=test_namespace, async_req=True)
            description1 = future1.result(timeout=30)
            assert description1.name == test_namespace

            # Try to create duplicate namespace - should raise an error
            future2 = idx.create_namespace(name=test_namespace, async_req=True)

            # GRPC errors are wrapped in PineconeException, not PineconeApiException
            from pinecone.exceptions import PineconeException

            with pytest.raises(PineconeException):
                future2.result(timeout=30)

        finally:
            # Cleanup
            if verify_namespace_exists(idx, test_namespace):
                idx.delete_namespace(namespace=test_namespace)

    def test_create_namespace_future_multiple(self, idx):
        """Test creating multiple namespaces asynchronously"""
        test_namespaces = [random_string(20) for i in range(3)]

        try:
            # Create all namespaces asynchronously
            futures = [idx.create_namespace(name=ns, async_req=True) for ns in test_namespaces]

            # Wait for all to complete
            from concurrent.futures import as_completed

            results = []
            for future in as_completed(futures, timeout=60):
                description = future.result()
                results.append(description)

            # Verify all were created
            assert len(results) == len(test_namespaces)
            namespace_names = [desc.name for desc in results]
            for test_ns in test_namespaces:
                assert test_ns in namespace_names

            # Verify each namespace exists
            for ns in test_namespaces:
                verify_description = idx.describe_namespace(namespace=ns)
                assert verify_description.name == ns

        finally:
            # Cleanup
            for ns in test_namespaces:
                if verify_namespace_exists(idx, ns):
                    idx.delete_namespace(namespace=ns)
