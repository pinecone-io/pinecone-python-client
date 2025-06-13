import pytest
import logging
from pinecone import Admin, Pinecone, NotFoundException
from datetime import datetime

logger = logging.getLogger(__name__)


class TestAdminProjects:
    def test_create_project(self):
        admin = Admin()
        project = admin.project.create(name="test-project")
        logger.info(f"Project created: {project}")

        try:
            assert project.name == "test-project"
            assert project.max_pods == 0
            assert project.force_encryption_with_cmek is False
            assert project.organization_id is not None
            assert isinstance(project.organization_id, str)
            assert project.created_at is not None
            assert isinstance(project.created_at, datetime)

            # Test dictionary-style access to project attributes
            assert project["name"] == "test-project"
            assert project["max_pods"] == 0
            assert project["force_encryption_with_cmek"] is False
            assert project["organization_id"] is not None
            assert isinstance(project["organization_id"], str)
            assert project["created_at"] is not None

            # Test get-style access to project attributes
            assert project.get("name") == "test-project"
            assert project.get("max_pods") == 0
            assert project.get("force_encryption_with_cmek") is False
            assert project.get("organization_id") is not None
            assert isinstance(project.get("organization_id"), str)
            assert project.get("created_at") is not None

            # Test projects can be listed. Combining this with the create
            # test means we can be assured there is at least one project
            project_list = admin.project.list().data
            logger.info(f"Projects: {project_list}")
            assert isinstance(project_list, list)
            assert len(project_list) > 0

            assert project_list[0].id is not None
            assert project_list[0].name is not None
            assert project_list[0].max_pods is not None
            assert project_list[0].force_encryption_with_cmek is not None
            assert project_list[0].organization_id is not None
            assert project_list[0].created_at is not None

            # Test that I can fetch the project I just created by id
            project_by_id = admin.project.get(project_id=project.id)
            assert project_by_id.id == project.id
            assert project_by_id.name == project.name

            # Test that I can fetch the project using aliased methods
            project_by_id_alt = admin.project.describe(project_id=project.id)
            assert project_by_id_alt.id == project.id
            assert project_by_id_alt.name == project.name

            project_by_name_alt2 = admin.project.fetch(project_id=project.id)
            assert project_by_name_alt2.id == project.id
            assert project_by_name_alt2.name == project.name

            # Test that I can fetch the project I just created by name
            project_by_name = admin.project.get(name=project.name)
            assert project_by_name.id == project.id
            assert project_by_name.name == project.name

            # Test that I can update the project
            updated = admin.project.update(
                project_id=project.id,
                name="test-project-updated",
                max_pods=1,
                force_encryption_with_cmek=True,
            )
            assert updated.id == project.id
            assert updated.name == "test-project-updated"
            assert updated.max_pods == 1
            assert updated.force_encryption_with_cmek is True
        finally:
            # Clean up
            admin.project.delete(project_id=project.id)
            logger.info(f"Project deleted: {project.id}")

            # Test that the project is deleted
            with pytest.raises(NotFoundException):
                admin.project.get(project_id=project.id)

    def test_delete_project_containing_indexes(self):
        admin = Admin()
        project = admin.project.create(name="test-project-with-stuff")
        logger.info(f"Project created: {project}")

        try:
            # Create an api key
            api_key = admin.api_key.create(project_id=project.id, name="test-api-key")
            logger.info(f"API key created: {api_key.key.id}")

            pc = Pinecone(api_key=api_key.value)
            created_index = pc.db.index.create(
                name="test-index",
                dimension=100,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
                deletion_protection="enabled",  # extra hard to delete
            )
            logger.info(f"Index created: {created_index.name}")

            # Delete the project
            with pytest.raises(Exception) as e:
                admin.project.delete(project_id=project.id, delete_all_indexes=True)
            assert "Indexes with deletion protection enabled cannot be deleted" in str(e)

            pc.db.index.configure(name=created_index.name, deletion_protection="disabled")

            admin.project.delete(project_id=project.id, delete_all_indexes=True)

            logger.info(f"Project deleted: {project.id}")
        finally:
            # Clean up
            if admin.project.exists(project_id=project.id):
                admin.project.delete(project_id=project.id, delete_all_indexes=True)
                logger.info(f"Project deleted: {project.id}")

            # Test that the project is deleted
            with pytest.raises(NotFoundException):
                admin.project.get(project_id=project.id)
