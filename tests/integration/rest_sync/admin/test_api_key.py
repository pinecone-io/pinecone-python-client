import logging

from pinecone import Admin

logger = logging.getLogger(__name__)


class TestAdminApiKey:
    def test_create_api_key(self):
        admin = Admin()
        project_name = "test-project-for-api-key"
        if not admin.project.exists(name=project_name):
            project = admin.project.create(name=project_name)
        else:
            project = admin.project.get(name=project_name)

        try:
            # Create an API key
            key_response = admin.api_key.create(project_id=project.id, name="test-api-key1")
            logger.info(f"API key created: {key_response.key.id}")

            assert key_response.key.created_at is not None
            assert key_response.key.id is not None
            assert isinstance(key_response.key.id, str)
            assert key_response.key.name == "test-api-key1"
            assert key_response.key.project_id == project.id
            assert key_response.key.roles[0] == "ProjectEditor"
            assert key_response.value is not None
            assert isinstance(key_response.value, str)

            # Create a second API key with non-default role
            key_response2 = admin.api_key.create(
                project_id=project.id, name="test-api-key2", roles=["ProjectViewer"]
            )
            logger.info(f"API key created: {key_response2.key.id}")

            assert key_response2.key.created_at is not None
            assert key_response2.key.id is not None
            assert isinstance(key_response2.key.id, str)
            assert key_response2.key.name == "test-api-key2"
            assert key_response2.key.project_id == project.id
            assert key_response2.key.roles[0] == "ProjectViewer"
            assert key_response2.value is not None
            assert isinstance(key_response2.value, str)

            # Verify dictionary-style access to key attributes
            assert key_response.key["created_at"] is not None
            assert key_response.key["id"] is not None
            assert isinstance(key_response.key["id"], str)
            assert key_response.key["name"] == "test-api-key1"
            assert key_response.key["project_id"] == project.id

            # Verify get-style access to key attributes
            assert key_response.key.get("created_at") is not None
            assert key_response.key.get("id") is not None
            assert isinstance(key_response.key.get("id"), str)
            assert key_response.key.get("name") == "test-api-key1"
            assert key_response.key.get("project_id") == project.id

            # Get a key by id
            key_response_by_id = admin.api_key.fetch(api_key_id=key_response.key.id)
            assert key_response_by_id.id == key_response.key.id
            assert key_response_by_id.name == key_response.key.name
            assert key_response_by_id.project_id == key_response.key.project_id
            assert key_response_by_id.roles == key_response.key.roles

            # List API keys
            key_list = admin.api_key.list(project_id=project.id).data
            assert isinstance(key_list, list)
            assert len(key_list) == 2
            ids = [key.id for key in key_list]
            assert key_response.key.id in ids
            assert key_response2.key.id in ids

            # Delete the first API key
            admin.api_key.delete(api_key_id=key_response.key.id)
            logger.info(f"API key deleted: {key_response.key.id}")

            # Verify key is deleted
            key_list = admin.api_key.list(project_id=project.id).data
            logger.info(f"API keys: {key_list}")
            assert isinstance(key_list, list)
            assert len(key_list) == 1
            key_list_ids = [key.id for key in key_list]
            assert key_response2.key.id in key_list_ids
            assert key_response.key.id not in key_list_ids

            # Delete the second API key
            admin.api_key.delete(api_key_id=key_response2.key.id)
            logger.info(f"API key deleted: {key_response2.key.id}")

            # Verify all keys are deleted
            key_list = admin.api_key.list(project_id=project.id).data
            logger.info(f"API keys: {key_list}")
            assert len(key_list) == 0
        finally:
            # Clean up
            admin.project.delete(project_id=project.id)
            logger.info(f"Project deleted: {project.id}")

    def test_fetch_aliases(self):
        admin = Admin()
        project_name = "test-project-for-api-key"
        if not admin.project.exists(name=project_name):
            project = admin.project.create(name=project_name)
        else:
            project = admin.project.get(name=project_name)

        try:
            # Create an API key
            key_response = admin.api_key.create(project_id=project.id, name="test-api-key1")

            # Fetch the API key using the aliases
            key_response_by_id = admin.api_key.fetch(api_key_id=key_response.key.id)
            logger.info(f"API key fetched with id: {key_response_by_id.id}")
            assert key_response_by_id.id == key_response.key.id

            get_key_response = admin.api_key.get(api_key_id=key_response.key.id)
            logger.info(f"API key fetched with id: {get_key_response.id}")
            assert get_key_response.id == key_response.key.id

            described_key_response = admin.api_key.describe(api_key_id=key_response.key.id)
            assert described_key_response.id == key_response.key.id

        finally:
            admin.project.delete(project_id=project.id)
            logger.info(f"Project deleted: {project.id}")

    def test_update_api_key(self):
        admin = Admin()
        project_name = "test-project-for-api-key-update"
        if not admin.project.exists(name=project_name):
            project = admin.project.create(name=project_name)
        else:
            project = admin.project.get(name=project_name)

        try:
            # Create an API key
            key_response = admin.api_key.create(
                project_id=project.id, name="test-api-key-update", roles=["ProjectEditor"]
            )
            logger.info(f"API key created: {key_response.key.id}")

            original_roles = key_response.key.roles

            # Update the API key's name
            updated_key = admin.api_key.update(
                api_key_id=key_response.key.id, name="test-api-key-updated-name"
            )
            logger.info(f"API key updated: {updated_key.id}")

            assert updated_key.id == key_response.key.id
            assert updated_key.name == "test-api-key-updated-name"
            assert updated_key.roles == original_roles  # Roles should not change

            # Update the API key's roles
            updated_key = admin.api_key.update(
                api_key_id=key_response.key.id, roles=["ProjectViewer"]
            )
            logger.info(f"API key roles updated: {updated_key.id}")

            assert updated_key.id == key_response.key.id
            assert updated_key.name == "test-api-key-updated-name"  # Name should not change
            assert updated_key.roles == ["ProjectViewer"]

            # Update both name and roles
            updated_key = admin.api_key.update(
                api_key_id=key_response.key.id,
                name="test-api-key-final",
                roles=["ProjectEditor", "DataPlaneEditor"],
            )
            logger.info(f"API key name and roles updated: {updated_key.id}")

            assert updated_key.id == key_response.key.id
            assert updated_key.name == "test-api-key-final"
            assert set(updated_key.roles) == set(["ProjectEditor", "DataPlaneEditor"])

            # Verify by fetching the key
            fetched_key = admin.api_key.fetch(api_key_id=key_response.key.id)
            assert fetched_key.name == "test-api-key-final"
            assert set(fetched_key.roles) == set(["ProjectEditor", "DataPlaneEditor"])

            # Clean up
            admin.api_key.delete(api_key_id=key_response.key.id)
            logger.info(f"API key deleted: {key_response.key.id}")

        finally:
            # Clean up project
            admin.project.delete(project_id=project.id)
            logger.info(f"Project deleted: {project.id}")
