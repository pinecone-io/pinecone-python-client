import pytest
from pinecone import IndexList
from pinecone.core.openapi.db_control.models import (
    IndexList as OpenApiIndexList,
    IndexModel as OpenApiIndexModel,
    IndexModelStatus,
)
from pinecone.core.openapi.db_control.model.schema import Schema
from pinecone.core.openapi.db_control.model.schema_fields import SchemaFields
from pinecone.core.openapi.db_control.model.deployment import Deployment


def _create_test_schema(dimension=10):
    """Create a test schema with a dense vector field."""
    return Schema(
        fields={"_values": SchemaFields(type="dense_vector", dimension=dimension, metric="cosine")},
        _check_type=False,
    )


def _create_test_pod_deployment():
    """Create a test pod deployment."""
    return Deployment(
        deployment_type="pod",
        environment="us-west1-gcp",
        pod_type="p1.x1",
        pods=1,
        replicas=1,
        shards=1,
        _check_type=False,
    )


@pytest.fixture
def index_list_response():
    """Fixture using alpha API structure with schema + deployment."""
    return OpenApiIndexList(
        indexes=[
            OpenApiIndexModel(
                name="test-index-1",
                schema=_create_test_schema(dimension=2),
                deployment=_create_test_pod_deployment(),
                host="https://test-index-1.pinecone.io",
                status=IndexModelStatus(ready=True, state="Ready", _check_type=False),
                deletion_protection="enabled",
                _check_type=False,
            ),
            OpenApiIndexModel(
                name="test-index-2",
                schema=_create_test_schema(dimension=3),
                deployment=_create_test_pod_deployment(),
                host="https://test-index-2.pinecone.io",
                status=IndexModelStatus(ready=True, state="Ready", _check_type=False),
                deletion_protection="disabled",
                _check_type=False,
            ),
        ],
        _check_type=False,
    )


class TestIndexList:
    def test_index_list_has_length(self, index_list_response):
        assert len(IndexList(index_list_response)) == 2

    def test_index_list_is(self, index_list_response):
        iil = IndexList(index_list_response)
        assert [i["name"] for i in iil] == ["test-index-1", "test-index-2"]
        # dimension and metric are accessed through compatibility layer
        assert [i.dimension for i in iil] == [2, 3]
        assert [i.metric for i in iil] == ["cosine", "cosine"]

    def test_index_list_names_syntactic_sugar(self, index_list_response):
        iil = IndexList(index_list_response)
        assert iil.names() == ["test-index-1", "test-index-2"]

    def test_index_list_getitem(self, index_list_response):
        iil = IndexList(index_list_response)
        input_list = index_list_response
        assert input_list.indexes[0].name == iil[0].name
        # Access dimension/metric through compatibility layer
        assert iil[0].dimension == 2
        assert iil[0].metric == "cosine"
        assert input_list.indexes[0].host == iil[0].host
        assert input_list.indexes[0].deletion_protection == iil[0].deletion_protection
        assert iil[0].deletion_protection == "enabled"

        assert input_list.indexes[1].name == iil[1].name

    def test_index_list_proxies_methods(self, index_list_response):
        # Forward compatibility, in case we add more attributes to IndexList for pagination
        assert IndexList(index_list_response).indexes[0].name == index_list_response.indexes[0].name

    def test_when_results_are_empty(self):
        iil = IndexList(OpenApiIndexList(indexes=[]))
        assert len(iil) == 0
        assert iil.index_list.indexes == []
        assert iil.indexes == []
        assert iil.names() == []
