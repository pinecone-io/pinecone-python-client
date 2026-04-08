"""Unit tests for summary __repr__ on list wrapper classes."""

from __future__ import annotations

from pinecone.models.admin.api_key import APIKeyList, APIKeyModel
from pinecone.models.admin.organization import OrganizationList, OrganizationModel
from pinecone.models.admin.project import ProjectList, ProjectModel
from pinecone.models.backups.list import BackupList, RestoreJobList
from pinecone.models.backups.model import BackupModel, RestoreJobModel
from pinecone.models.collections.list import CollectionList
from pinecone.models.collections.model import CollectionModel
from pinecone.models.imports.list import ImportList
from pinecone.models.imports.model import ImportModel
from pinecone.models.inference.model_list import ModelInfoList
from pinecone.models.inference.models import ModelInfo


class TestCollectionListRepr:
    def test_repr_summary_format(self) -> None:
        col = CollectionModel(name="col1", status="Ready", environment="us-east1-gcp")
        cl = CollectionList([col])
        r = repr(cl)
        assert r == "CollectionList([<name='col1', status='Ready'>])"

    def test_repr_multiple_items(self) -> None:
        cols = [
            CollectionModel(name="col1", status="Ready", environment="us-east1-gcp"),
            CollectionModel(name="col2", status="Initializing", environment="us-east1-gcp"),
        ]
        cl = CollectionList(cols)
        r = repr(cl)
        assert "CollectionList([" in r
        assert "<name='col1', status='Ready'>" in r
        assert "<name='col2', status='Initializing'>" in r
        assert "CollectionModel" not in r

    def test_repr_empty_list(self) -> None:
        cl = CollectionList([])
        assert repr(cl) == "CollectionList([])"


class TestBackupListRepr:
    def _make_backup(
        self,
        backup_id: str = "bkp-abc",
        name: str | None = "daily",
        status: str = "Ready",
        source: str = "my-index",
    ) -> BackupModel:
        return BackupModel(
            backup_id=backup_id,
            source_index_name=source,
            source_index_id="idx-xyz",
            status=status,
            cloud="aws",
            region="us-east-1",
            name=name,
        )

    def test_repr_with_name(self) -> None:
        bl = BackupList([self._make_backup()])
        r = repr(bl)
        assert "BackupList([" in r
        assert "<name='daily', status='Ready', source='my-index'>" in r
        assert "BackupModel" not in r

    def test_repr_falls_back_to_backup_id_when_no_name(self) -> None:
        bl = BackupList([self._make_backup(name=None)])
        r = repr(bl)
        assert "'bkp-abc'" in r

    def test_repr_empty_list(self) -> None:
        bl = BackupList([])
        assert repr(bl) == "BackupList([])"


class TestRestoreJobListRepr:
    def _make_job(
        self,
        restore_job_id: str = "rj-abc",
        status: str = "Completed",
        target: str = "my-index",
    ) -> RestoreJobModel:
        return RestoreJobModel(
            restore_job_id=restore_job_id,
            backup_id="bkp-xyz",
            target_index_name=target,
            target_index_id="idx-xyz",
            status=status,
            created_at="2025-01-01T00:00:00Z",
        )

    def test_repr_summary_format(self) -> None:
        rl = RestoreJobList([self._make_job()])
        r = repr(rl)
        assert "RestoreJobList([" in r
        assert "<id='rj-abc', status='Completed', target='my-index'>" in r
        assert "RestoreJobModel" not in r

    def test_repr_empty_list(self) -> None:
        rl = RestoreJobList([])
        assert repr(rl) == "RestoreJobList([])"


class TestImportListRepr:
    def _make_import(
        self,
        id: str = "imp-abc",
        status: str = "InProgress",
        percent: float | None = 45.0,
    ) -> ImportModel:
        return ImportModel(
            id=id,
            uri="s3://bucket/path",
            status=status,
            created_at="2025-01-01T00:00:00Z",
            percent_complete=percent,
        )

    def test_repr_summary_format(self) -> None:
        il = ImportList([self._make_import()])
        r = repr(il)
        assert "ImportList([" in r
        assert "<id='imp-abc', status='InProgress', percent=45.0>" in r
        assert "ImportModel" not in r

    def test_repr_percent_none(self) -> None:
        il = ImportList([self._make_import(percent=None)])
        r = repr(il)
        assert "percent=None" in r

    def test_repr_empty_list(self) -> None:
        il = ImportList([])
        assert repr(il) == "ImportList([])"


class TestModelInfoListRepr:
    def _make_model(self, model: str = "multilingual-e5-large", type: str = "embed") -> ModelInfo:
        return ModelInfo(
            model=model,
            short_description="Test model",
            type=type,
            supported_parameters=[],
        )

    def test_repr_summary_format(self) -> None:
        ml = ModelInfoList([self._make_model()])
        r = repr(ml)
        assert "ModelInfoList([" in r
        assert "<model='multilingual-e5-large', type='embed'>" in r
        assert "ModelInfo(" not in r

    def test_repr_multiple_items(self) -> None:
        ml = ModelInfoList(
            [
                self._make_model("multilingual-e5-large", "embed"),
                self._make_model("pinecone-rerank-v0", "rerank"),
            ]
        )
        r = repr(ml)
        assert "<model='multilingual-e5-large', type='embed'>" in r
        assert "<model='pinecone-rerank-v0', type='rerank'>" in r

    def test_repr_empty_list(self) -> None:
        ml = ModelInfoList([])
        assert repr(ml) == "ModelInfoList([])"


class TestAPIKeyListRepr:
    def _make_key(self, name: str = "prod-key", project_id: str = "proj-abc") -> APIKeyModel:
        return APIKeyModel(id="key-xyz", name=name, project_id=project_id, roles=["IndexReadWrite"])

    def test_repr_summary_format(self) -> None:
        kl = APIKeyList([self._make_key()])
        r = repr(kl)
        assert "APIKeyList([" in r
        assert "<name='prod-key', project_id='proj-abc'>" in r
        assert "APIKeyModel" not in r

    def test_repr_empty_list(self) -> None:
        kl = APIKeyList([])
        assert repr(kl) == "APIKeyList([])"


class TestOrganizationListRepr:
    def _make_org(self, name: str = "acme-corp", plan: str = "Enterprise") -> OrganizationModel:
        return OrganizationModel(
            id="org-xyz",
            name=name,
            plan=plan,
            payment_status="active",
            created_at="2025-01-01T00:00:00Z",
            support_tier="enterprise",
        )

    def test_repr_summary_format(self) -> None:
        ol = OrganizationList([self._make_org()])
        r = repr(ol)
        assert "OrganizationList([" in r
        assert "<name='acme-corp', plan='Enterprise'>" in r
        assert "OrganizationModel" not in r

    def test_repr_empty_list(self) -> None:
        ol = OrganizationList([])
        assert repr(ol) == "OrganizationList([])"


class TestProjectListRepr:
    def _make_project(self, name: str = "production", id: str = "proj-abc") -> ProjectModel:
        return ProjectModel(
            id=id,
            name=name,
            max_pods=5,
            force_encryption_with_cmek=False,
            organization_id="org-xyz",
        )

    def test_repr_summary_format(self) -> None:
        pl = ProjectList([self._make_project()])
        r = repr(pl)
        assert "ProjectList([" in r
        assert "<name='production', id='proj-abc'>" in r
        assert "ProjectModel" not in r

    def test_repr_empty_list(self) -> None:
        pl = ProjectList([])
        assert repr(pl) == "ProjectList([])"
