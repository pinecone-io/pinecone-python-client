"""Unit tests for OrganizationResource delete method.

These tests verify that the delete() method correctly builds and passes requests
to the underlying API client without making real API calls.
"""

import pytest

from pinecone.admin.resources.organization import OrganizationResource
from pinecone.openapi_support import ApiClient


class TestOrganizationResourceDelete:
    """Test parameter translation in OrganizationResource.delete()"""

    def setup_method(self):
        """Set up test fixtures"""
        api_client = ApiClient()
        self.organization_resource = OrganizationResource(api_client=api_client)

    def test_delete_calls_api_with_organization_id(self, mocker):
        """Test delete() calls the API method with correct organization_id"""
        mocker.patch.object(
            self.organization_resource._organizations_api, "delete_organization", autospec=True
        )

        organization_id = "test-org-id-123"
        self.organization_resource.delete(organization_id=organization_id)

        # Verify API was called with correct arguments
        self.organization_resource._organizations_api.delete_organization.assert_called_once_with(
            organization_id=organization_id
        )

    def test_delete_requires_organization_id(self):
        """Test that delete() requires organization_id parameter via @require_kwargs"""
        with pytest.raises(TypeError):
            self.organization_resource.delete()

    def test_delete_with_different_organization_id(self, mocker):
        """Test delete() with a different organization_id value"""
        mocker.patch.object(
            self.organization_resource._organizations_api, "delete_organization", autospec=True
        )

        organization_id = "another-org-id-456"
        self.organization_resource.delete(organization_id=organization_id)

        # Verify API was called with the specific organization_id
        self.organization_resource._organizations_api.delete_organization.assert_called_once_with(
            organization_id=organization_id
        )
