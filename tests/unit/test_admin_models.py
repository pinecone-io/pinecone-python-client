"""Tests for Admin API response models."""

from __future__ import annotations

import pytest

from pinecone.models.admin.api_key import APIKeyList, APIKeyModel, APIKeyWithSecret
from pinecone.models.admin.organization import OrganizationList, OrganizationModel
from pinecone.models.admin.project import ProjectList, ProjectModel


class TestOrganizationModel:
    def test_organization_model_fields(self) -> None:
        org = OrganizationModel(
            id="org-123",
            name="My Org",
            plan="Enterprise",
            payment_status="Active",
            created_at="2025-01-01T00:00:00Z",
            support_tier="Basic",
        )
        assert org.id == "org-123"
        assert org.name == "My Org"
        assert org.plan == "Enterprise"
        assert org.payment_status == "Active"
        assert org.created_at == "2025-01-01T00:00:00Z"
        assert org.support_tier == "Basic"

    def test_organization_model_support_tier_field(self) -> None:
        org = OrganizationModel(
            id="org-123",
            name="My Org",
            plan="Enterprise",
            payment_status="Active",
            created_at="2025-01-01T00:00:00Z",
            support_tier="Premium",
        )
        assert org.support_tier == "Premium"
        assert org["support_tier"] == "Premium"

    def test_organization_model_bracket_access(self) -> None:
        org = OrganizationModel(
            id="org-123",
            name="My Org",
            plan="Free",
            payment_status="Active",
            created_at="2025-01-01T00:00:00Z",
            support_tier="Basic",
        )
        assert org["name"] == "My Org"
        assert org["id"] == "org-123"
        with pytest.raises(KeyError, match="missing"):
            org["missing"]

    def test_organization_list_names(self) -> None:
        kwargs = {
            "plan": "Free",
            "payment_status": "Active",
            "created_at": "2025-01-01T00:00:00Z",
            "support_tier": "Basic",
        }
        orgs = OrganizationList(
            [
                OrganizationModel(id="org-1", name="Alpha", **kwargs),
                OrganizationModel(id="org-2", name="Beta", **kwargs),
                OrganizationModel(id="org-3", name="Gamma", **kwargs),
            ]
        )
        assert orgs.names() == ["Alpha", "Beta", "Gamma"]

    def test_organization_list_len_and_iter(self) -> None:
        kwargs = {
            "plan": "Free",
            "payment_status": "Active",
            "created_at": "2025-01-01T00:00:00Z",
            "support_tier": "Basic",
        }
        orgs = OrganizationList(
            [
                OrganizationModel(id="org-1", name="Alpha", **kwargs),
                OrganizationModel(id="org-2", name="Beta", **kwargs),
            ]
        )
        assert len(orgs) == 2
        names = [o.name for o in orgs]
        assert names == ["Alpha", "Beta"]

    def test_organization_list_getitem(self) -> None:
        org = OrganizationModel(
            id="org-1",
            name="Alpha",
            plan="Free",
            payment_status="Active",
            created_at="2025-01-01T00:00:00Z",
            support_tier="Basic",
        )
        orgs = OrganizationList([org])
        assert orgs[0] is org


class TestProjectModel:
    def test_project_model_fields(self) -> None:
        project = ProjectModel(
            id="proj-123",
            name="My Project",
            max_pods=5,
            force_encryption_with_cmek=True,
            organization_id="org-456",
            created_at="2025-01-01T00:00:00Z",
        )
        assert project.id == "proj-123"
        assert project.name == "My Project"
        assert project.max_pods == 5
        assert project.force_encryption_with_cmek is True
        assert project.organization_id == "org-456"
        assert project.created_at == "2025-01-01T00:00:00Z"

    def test_project_model_optional_defaults(self) -> None:
        project = ProjectModel(
            id="proj-123",
            name="My Project",
            max_pods=0,
            force_encryption_with_cmek=False,
            organization_id="org-456",
        )
        assert project.created_at is None

    def test_project_model_bracket_access(self) -> None:
        project = ProjectModel(
            id="proj-123",
            name="My Project",
            max_pods=0,
            force_encryption_with_cmek=False,
            organization_id="org-456",
        )
        assert project["name"] == "My Project"
        with pytest.raises(KeyError, match="missing"):
            project["missing"]

    def test_project_list_names(self) -> None:
        projects = ProjectList(
            [
                ProjectModel(
                    id="p-1",
                    name="Proj A",
                    max_pods=0,
                    force_encryption_with_cmek=False,
                    organization_id="org-1",
                ),
                ProjectModel(
                    id="p-2",
                    name="Proj B",
                    max_pods=5,
                    force_encryption_with_cmek=True,
                    organization_id="org-1",
                ),
            ]
        )
        assert projects.names() == ["Proj A", "Proj B"]

    def test_project_list_len_and_iter(self) -> None:
        projects = ProjectList(
            [
                ProjectModel(
                    id="p-1",
                    name="Proj A",
                    max_pods=0,
                    force_encryption_with_cmek=False,
                    organization_id="org-1",
                ),
            ]
        )
        assert len(projects) == 1
        assert [p.name for p in projects] == ["Proj A"]


class TestAPIKeyModel:
    def test_api_key_model_fields(self) -> None:
        api_key = APIKeyModel(
            id="key-123",
            name="my-key",
            project_id="proj-456",
            roles=["ProjectEditor", "DataPlaneEditor"],
        )
        assert api_key.id == "key-123"
        assert api_key.name == "my-key"
        assert api_key.project_id == "proj-456"
        assert api_key.roles == ["ProjectEditor", "DataPlaneEditor"]

    def test_api_key_model_bracket_access(self) -> None:
        api_key = APIKeyModel(
            id="key-123",
            name="my-key",
            project_id="proj-456",
            roles=["ProjectEditor"],
        )
        assert api_key["name"] == "my-key"
        assert api_key["roles"] == ["ProjectEditor"]
        with pytest.raises(KeyError, match="missing"):
            api_key["missing"]

    def test_api_key_with_secret(self) -> None:
        inner_key = APIKeyModel(
            id="key-123",
            name="my-key",
            project_id="proj-456",
            roles=["ProjectEditor"],
        )
        secret = APIKeyWithSecret(key=inner_key, value="pcsk_secret_value")
        assert secret.key is inner_key
        assert secret.value == "pcsk_secret_value"
        assert secret.key.name == "my-key"

    def test_api_key_with_secret_bracket_access(self) -> None:
        inner_key = APIKeyModel(
            id="key-123",
            name="my-key",
            project_id="proj-456",
            roles=["ProjectEditor"],
        )
        secret = APIKeyWithSecret(key=inner_key, value="pcsk_secret_value")
        assert secret["value"] == "pcsk_secret_value"
        assert secret["key"] is inner_key
        with pytest.raises(KeyError, match="missing"):
            secret["missing"]

    def test_api_key_with_secret_repr_masks_value(self) -> None:
        inner_key = APIKeyModel(
            id="key-123",
            name="my-key",
            project_id="proj-456",
            roles=["ProjectEditor"],
        )
        secret = APIKeyWithSecret(key=inner_key, value="pcsk_abc123_secret_key9")
        result = repr(secret)
        assert "pcsk_abc123_secret_key9" not in result
        assert "...key9'" in result
        assert "APIKeyWithSecret(" in result
        assert "key=" in result

    def test_api_key_with_secret_repr_masks_short_value(self) -> None:
        inner_key = APIKeyModel(
            id="key-123",
            name="my-key",
            project_id="proj-456",
            roles=["ProjectEditor"],
        )
        secret = APIKeyWithSecret(key=inner_key, value="ab")
        result = repr(secret)
        assert "ab" not in result or "***" in result
        assert "***" in result

    def test_api_key_list_names(self) -> None:
        keys = APIKeyList(
            [
                APIKeyModel(id="k-1", name="key-alpha", project_id="p-1", roles=["ProjectEditor"]),
                APIKeyModel(id="k-2", name="key-beta", project_id="p-1", roles=["ProjectViewer"]),
            ]
        )
        assert keys.names() == ["key-alpha", "key-beta"]

    def test_api_key_list_len_and_iter(self) -> None:
        keys = APIKeyList(
            [
                APIKeyModel(id="k-1", name="key-alpha", project_id="p-1", roles=["ProjectEditor"]),
            ]
        )
        assert len(keys) == 1
        assert [k.name for k in keys] == ["key-alpha"]
