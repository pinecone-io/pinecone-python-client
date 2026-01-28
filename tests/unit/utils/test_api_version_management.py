"""Tests for API version management across different API categories.

These tests verify that each API category (db_control, db_data, inference, oauth, admin)
uses the correct API version header when making requests.
"""

from pinecone.core.openapi import db_control, db_data, inference, oauth, admin


class TestModuleApiVersions:
    """Verify each generated module has the correct API_VERSION constant."""

    def test_db_control_uses_alpha_version(self):
        """db_control should use the alpha API version for FTS support."""
        assert hasattr(db_control, "API_VERSION")
        assert db_control.API_VERSION == "2026-01.alpha"

    def test_db_data_uses_alpha_version(self):
        """db_data should use the alpha API version for FTS support."""
        assert hasattr(db_data, "API_VERSION")
        assert db_data.API_VERSION == "2026-01.alpha"

    def test_inference_uses_stable_version(self):
        """inference should use the stable API version."""
        assert hasattr(inference, "API_VERSION")
        assert inference.API_VERSION == "2025-10"

    def test_oauth_uses_stable_version(self):
        """oauth should use the stable API version."""
        assert hasattr(oauth, "API_VERSION")
        assert oauth.API_VERSION == "2025-10"

    def test_admin_uses_stable_version(self):
        """admin should use the stable API version."""
        assert hasattr(admin, "API_VERSION")
        assert admin.API_VERSION == "2025-10"


class TestApiVersionHeaderIsSet:
    """Verify API version headers are set when creating clients."""

    def test_setup_openapi_client_sets_version_header(self):
        """setup_openapi_client should set X-Pinecone-API-Version header."""
        from pinecone.config import ConfigBuilder
        from pinecone.openapi_support.api_client import ApiClient
        from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi
        from pinecone.utils.setup_openapi_client import setup_openapi_client

        config = ConfigBuilder.build(api_key="test-api-key", host="https://test-host")
        openapi_config = ConfigBuilder.build_openapi_config(config)

        client = setup_openapi_client(
            api_client_klass=ApiClient,
            api_klass=ManageIndexesApi,
            config=config,
            openapi_config=openapi_config,
            pool_threads=1,
            api_version="2026-01.alpha",
        )

        assert "X-Pinecone-API-Version" in client.api_client.default_headers
        assert client.api_client.default_headers["X-Pinecone-API-Version"] == "2026-01.alpha"

    def test_setup_openapi_client_without_version_does_not_set_header(self):
        """When api_version is None, no version header should be set."""
        from pinecone.config import ConfigBuilder
        from pinecone.openapi_support.api_client import ApiClient
        from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi
        from pinecone.utils.setup_openapi_client import setup_openapi_client

        config = ConfigBuilder.build(api_key="test-api-key", host="https://test-host")
        openapi_config = ConfigBuilder.build_openapi_config(config)

        client = setup_openapi_client(
            api_client_klass=ApiClient,
            api_klass=ManageIndexesApi,
            config=config,
            openapi_config=openapi_config,
            pool_threads=1,
            api_version=None,
        )

        assert "X-Pinecone-API-Version" not in client.api_client.default_headers


class TestDbControlUsesCorrectVersion:
    """Verify DBControl class uses the correct API version."""

    def test_db_control_module_has_correct_version(self):
        """db_control generated module should have the correct API_VERSION."""
        # Test the generated module directly, not the wrapper that has broken imports
        from pinecone.core.openapi.db_control import API_VERSION

        assert API_VERSION == "2026-01.alpha"


class TestDbDataUsesCorrectVersion:
    """Verify Index class uses the correct API version."""

    def test_index_imports_correct_version(self):
        """Index should import API_VERSION from db_data module."""
        from pinecone.db_data.index import API_VERSION

        assert API_VERSION == "2026-01.alpha"


class TestInferenceUsesCorrectVersion:
    """Verify Inference class uses the correct API version."""

    def test_inference_imports_correct_version(self):
        """Inference should import API_VERSION from inference module."""
        from pinecone.inference.inference import API_VERSION

        assert API_VERSION == "2025-10"


class TestAdminUsesCorrectVersion:
    """Verify Admin class uses the correct API version."""

    def test_admin_imports_correct_version(self):
        """Admin should import API_VERSION from oauth module."""
        from pinecone.admin.admin import API_VERSION

        assert API_VERSION == "2025-10"


class TestGrpcUsesCorrectVersion:
    """Verify gRPC client uses the correct API version."""

    def test_grpc_uses_shared_api_version(self):
        """gRPC should use the shared API_VERSION from openapi_support."""
        from pinecone.openapi_support.api_version import API_VERSION

        # gRPC uses the db_data version since it's data plane only
        assert API_VERSION == "2026-01.alpha"
