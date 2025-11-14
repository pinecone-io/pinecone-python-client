import logging
from datetime import datetime

from pinecone import Admin

logger = logging.getLogger(__name__)


class TestAdminOrganization:
    def test_update_organization(self):
        admin = Admin()

        # Get the current organization (usually there's only one)
        organizations_response = admin.organization.list()
        assert len(organizations_response.data) > 0, "No organizations found"

        organization = organizations_response.data[0]
        original_name = organization.name
        organization_id = organization.id

        logger.info(f"Original organization name: {original_name}")
        logger.info(f"Organization ID: {organization_id}")

        try:
            # Update the organization name
            updated_organization = admin.organization.update(
                organization_id=organization_id, name=f"{original_name}-updated-test"
            )
            logger.info(f"Organization updated: {updated_organization.name}")

            assert updated_organization.id == organization_id
            assert updated_organization.name == f"{original_name}-updated-test"

            # Verify by fetching the organization
            fetched_organization = admin.organization.fetch(organization_id=organization_id)
            assert fetched_organization.name == f"{original_name}-updated-test"

            # Revert the name change
            reverted_organization = admin.organization.update(
                organization_id=organization_id, name=original_name
            )
            logger.info(f"Organization name reverted: {reverted_organization.name}")

            assert reverted_organization.name == original_name

            # Verify the revert
            final_organization = admin.organization.fetch(organization_id=organization_id)
            assert final_organization.name == original_name

        except Exception as e:
            # If something goes wrong, try to revert the name
            logger.error(f"Error during test: {e}")
            try:
                admin.organization.update(organization_id=organization_id, name=original_name)
            except Exception as revert_error:
                logger.error(f"Failed to revert organization name: {revert_error}")
            raise

    def test_list_organizations(self):
        admin = Admin()

        # List all organizations
        organizations_response = admin.organization.list()
        logger.info(f"Organizations response: {organizations_response}")

        # Verify response structure
        assert hasattr(organizations_response, "data")
        assert isinstance(organizations_response.data, list)
        assert len(organizations_response.data) > 0, "No organizations found"

        # Verify first organization has all required fields
        org = organizations_response.data[0]
        logger.info(f"Organization: {org}")

        assert org.id is not None
        assert isinstance(org.id, str)
        assert org.name is not None
        assert isinstance(org.name, str)
        assert org.plan is not None
        assert isinstance(org.plan, str)
        assert org.payment_status is not None
        assert isinstance(org.payment_status, str)
        assert org.created_at is not None
        assert isinstance(org.created_at, datetime)
        assert org.support_tier is not None
        assert isinstance(org.support_tier, str)

        # Test dictionary-style access
        assert org["id"] is not None
        assert isinstance(org["id"], str)
        assert org["name"] is not None
        assert isinstance(org["name"], str)
        assert org["plan"] is not None
        assert isinstance(org["plan"], str)
        assert org["payment_status"] is not None
        assert isinstance(org["payment_status"], str)
        assert org["created_at"] is not None
        assert isinstance(org["created_at"], datetime)
        assert org["support_tier"] is not None
        assert isinstance(org["support_tier"], str)

        # Test get-style access
        assert org.get("id") is not None
        assert isinstance(org.get("id"), str)
        assert org.get("name") is not None
        assert isinstance(org.get("name"), str)
        assert org.get("plan") is not None
        assert isinstance(org.get("plan"), str)
        assert org.get("payment_status") is not None
        assert isinstance(org.get("payment_status"), str)
        assert org.get("created_at") is not None
        assert isinstance(org.get("created_at"), datetime)
        assert org.get("support_tier") is not None
        assert isinstance(org.get("support_tier"), str)

    def test_fetch_organization(self):
        admin = Admin()

        # First list organizations to get an organization_id
        organizations_response = admin.organization.list()
        assert len(organizations_response.data) > 0, "No organizations found"

        organization_id = organizations_response.data[0].id
        logger.info(f"Fetching organization: {organization_id}")

        # Fetch the organization by ID
        fetched_organization = admin.organization.fetch(organization_id=organization_id)
        logger.info(f"Fetched organization: {fetched_organization}")

        # Verify it matches the one from list
        listed_org = organizations_response.data[0]
        assert fetched_organization.id == listed_org.id
        assert fetched_organization.name == listed_org.name
        assert fetched_organization.plan == listed_org.plan
        assert fetched_organization.payment_status == listed_org.payment_status
        assert fetched_organization.created_at == listed_org.created_at
        assert fetched_organization.support_tier == listed_org.support_tier

        # Verify all fields are present and have correct types
        assert fetched_organization.id is not None
        assert isinstance(fetched_organization.id, str)
        assert fetched_organization.name is not None
        assert isinstance(fetched_organization.name, str)
        assert fetched_organization.plan is not None
        assert isinstance(fetched_organization.plan, str)
        assert fetched_organization.payment_status is not None
        assert isinstance(fetched_organization.payment_status, str)
        assert fetched_organization.created_at is not None
        assert isinstance(fetched_organization.created_at, datetime)
        assert fetched_organization.support_tier is not None
        assert isinstance(fetched_organization.support_tier, str)

        # Test dictionary-style access
        assert fetched_organization["id"] == organization_id
        assert fetched_organization["name"] is not None
        assert fetched_organization["plan"] is not None
        assert fetched_organization["payment_status"] is not None
        assert fetched_organization["created_at"] is not None
        assert fetched_organization["support_tier"] is not None

        # Test get-style access
        assert fetched_organization.get("id") == organization_id
        assert fetched_organization.get("name") is not None
        assert fetched_organization.get("plan") is not None
        assert fetched_organization.get("payment_status") is not None
        assert fetched_organization.get("created_at") is not None
        assert fetched_organization.get("support_tier") is not None

    def test_fetch_aliases(self):
        admin = Admin()

        # List organizations to get an organization_id
        organizations_response = admin.organization.list()
        assert len(organizations_response.data) > 0, "No organizations found"

        organization_id = organizations_response.data[0].id
        logger.info(f"Testing aliases for organization: {organization_id}")

        # Fetch the organization using fetch()
        fetched_org = admin.organization.fetch(organization_id=organization_id)
        logger.info(f"Organization by fetch: {fetched_org}")

        # Fetch the organization using get() alias
        get_org = admin.organization.get(organization_id=organization_id)
        logger.info(f"Organization by get: {get_org}")

        # Fetch the organization using describe() alias
        describe_org = admin.organization.describe(organization_id=organization_id)
        logger.info(f"Organization by describe: {describe_org}")

        # Verify all three methods return the same organization
        assert fetched_org.id == get_org.id == describe_org.id
        assert fetched_org.name == get_org.name == describe_org.name
        assert fetched_org.plan == get_org.plan == describe_org.plan
        assert fetched_org.payment_status == get_org.payment_status == describe_org.payment_status
        assert fetched_org.created_at == get_org.created_at == describe_org.created_at
        assert fetched_org.support_tier == get_org.support_tier == describe_org.support_tier
