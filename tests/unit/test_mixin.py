"""Regression tests for DictLikeStruct/StructDictMixin repr behavior.

Asserts that removing the mixin's __repr__ override restores msgspec's
default ClassName(field=value, ...) format.
"""

from __future__ import annotations

from pinecone.models.admin.api_key import APIKeyModel
from pinecone.models.backups.model import CreateIndexFromBackupResponse
from pinecone.models.response_info import ResponseInfo
from pinecone.models.vectors.responses import ListResponse


class TestMixinRepr:
    def test_list_response_repr_starts_with_class_name(self) -> None:
        obj = ListResponse()
        r = repr(obj)
        assert r.startswith("ListResponse("), f"Expected 'ListResponse(' prefix, got: {r!r}"
        assert not r.startswith("{'"), f"repr must not be dict form, got: {r!r}"

    def test_response_info_repr_starts_with_class_name(self) -> None:
        obj = ResponseInfo()
        r = repr(obj)
        assert r.startswith("ResponseInfo("), f"Expected 'ResponseInfo(' prefix, got: {r!r}"
        assert not r.startswith("{'"), f"repr must not be dict form, got: {r!r}"

    def test_create_index_from_backup_response_repr_starts_with_class_name(self) -> None:
        obj = CreateIndexFromBackupResponse(restore_job_id="rj-1", index_id="idx-1")
        r = repr(obj)
        assert r.startswith("CreateIndexFromBackupResponse("), (
            f"Expected 'CreateIndexFromBackupResponse(' prefix, got: {r!r}"
        )
        assert not r.startswith("{'"), f"repr must not be dict form, got: {r!r}"

    def test_api_key_model_repr_starts_with_class_name(self) -> None:
        obj = APIKeyModel(id="k1", name="n", project_id="p", roles=[], description=None)
        r = repr(obj)
        assert r.startswith("APIKeyModel("), f"Expected 'APIKeyModel(' prefix, got: {r!r}"
        assert not r.startswith("{'"), f"repr must not be dict form, got: {r!r}"

    def test_to_dict_still_returns_plain_dict(self) -> None:
        obj = ListResponse()
        d = obj.to_dict()
        assert isinstance(d, dict), "to_dict() must still return a plain dict"
        assert "vectors" in d

    def test_str_falls_back_to_repr(self) -> None:
        obj = ResponseInfo()
        assert str(obj) == repr(obj)
