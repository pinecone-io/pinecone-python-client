"""Tests for to_dict() on admin model classes."""

from __future__ import annotations

from pinecone.models.admin.api_key import APIKeyModel, APIKeyRole, APIKeyWithSecret
from pinecone.models.admin.organization import OrganizationModel
from pinecone.models.admin.project import ProjectModel


class TestAPIKeyModelToDict:
    def test_api_key_model_to_dict_required_fields(self) -> None:
        key = APIKeyModel(
            id="k1", name="prod-key", project_id="p1", roles=[APIKeyRole.PROJECT_EDITOR]
        )
        result = key.to_dict()
        assert isinstance(result, dict)
        assert "id" in result
        assert "name" in result
        assert "project_id" in result
        assert "roles" in result
        assert "description" in result
        assert result["id"] == "k1"
        assert result["name"] == "prod-key"
        assert result["project_id"] == "p1"

    def test_api_key_model_to_dict_roles_as_strings(self) -> None:
        key = APIKeyModel(
            id="k1", name="prod-key", project_id="p1", roles=[APIKeyRole.PROJECT_EDITOR]
        )
        result = key.to_dict()
        assert result["roles"] == ["ProjectEditor"]
        assert all(isinstance(r, str) for r in result["roles"])

    def test_api_key_model_to_dict_description_none(self) -> None:
        key = APIKeyModel(id="k1", name="prod-key", project_id="p1", roles=[])
        result = key.to_dict()
        assert result["description"] is None

    def test_api_key_model_to_dict_with_description(self) -> None:
        key = APIKeyModel(
            id="k1",
            name="prod-key",
            project_id="p1",
            roles=[APIKeyRole.DATA_PLANE_VIEWER],
            description="Used by search service",
        )
        result = key.to_dict()
        assert result["description"] == "Used by search service"


class TestAPIKeyWithSecretToDict:
    def test_api_key_with_secret_to_dict_nested(self) -> None:
        inner = APIKeyModel(
            id="k1", name="prod-key", project_id="p1", roles=[APIKeyRole.PROJECT_EDITOR]
        )
        secret = APIKeyWithSecret(key=inner, value="sk-abc")
        result = secret.to_dict()
        assert isinstance(result, dict)
        assert isinstance(result["key"], dict)
        assert not isinstance(result["key"], APIKeyModel)
        assert result["key"]["id"] == "k1"
        assert result["value"] == "sk-abc"


class TestOrganizationModelToDict:
    def test_organization_model_to_dict(self) -> None:
        org = OrganizationModel(
            id="org-1",
            name="acme-corp",
            plan="Standard",
            payment_status="active",
            created_at="2024-01-01T00:00:00Z",
            support_tier="basic",
        )
        result = org.to_dict()
        assert isinstance(result, dict)
        assert result["id"] == "org-1"
        assert result["name"] == "acme-corp"
        assert result["plan"] == "Standard"
        assert result["payment_status"] == "active"
        assert result["created_at"] == "2024-01-01T00:00:00Z"
        assert result["support_tier"] == "basic"


class TestProjectModelToDict:
    def test_project_model_to_dict(self) -> None:
        project = ProjectModel(
            id="proj-1",
            name="production-search",
            max_pods=5,
            force_encryption_with_cmek=False,
            organization_id="org-1",
        )
        result = project.to_dict()
        assert isinstance(result, dict)
        assert result["id"] == "proj-1"
        assert result["name"] == "production-search"
        assert result["max_pods"] == 5
        assert result["force_encryption_with_cmek"] is False
        assert result["organization_id"] == "org-1"
        assert "created_at" in result

    def test_project_model_to_dict_created_at_none(self) -> None:
        project = ProjectModel(
            id="proj-1",
            name="production-search",
            max_pods=5,
            force_encryption_with_cmek=False,
            organization_id="org-1",
        )
        result = project.to_dict()
        assert result["created_at"] is None


class TestToDictIsPureRead:
    def test_api_key_model_to_dict_is_pure_read(self) -> None:
        key = APIKeyModel(id="k1", name="prod-key", project_id="p1", roles=[])
        result = key.to_dict()
        result["name"] = "mutated"
        assert key.name == "prod-key"

    def test_api_key_with_secret_to_dict_is_pure_read(self) -> None:
        inner = APIKeyModel(id="k1", name="prod-key", project_id="p1", roles=[])
        secret = APIKeyWithSecret(key=inner, value="sk-abc")
        result = secret.to_dict()
        result["value"] = "mutated"
        assert secret.value == "sk-abc"
