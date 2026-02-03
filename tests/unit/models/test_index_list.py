import pytest
from pinecone import IndexList
from pinecone.core.openapi.db_control.models import IndexList as OpenApiIndexList, IndexModelStatus
from tests.fixtures import make_index_model


@pytest.fixture
def index_list_response():
    return OpenApiIndexList(
        indexes=[
            make_index_model(
                name="test-index-1",
                dimension=2,
                metric="cosine",
                host="https://test-index-1.pinecone.io",
                status=IndexModelStatus(ready=True, state="Ready"),
                deletion_protection="enabled",
                spec={
                    "pod": {
                        "environment": "us-west1-gcp",
                        "pod_type": "p1.x1",
                        "pods": 1,
                        "replicas": 1,
                        "shards": 1,
                    }
                },
            ).index,
            make_index_model(
                name="test-index-2",
                dimension=3,
                metric="cosine",
                host="https://test-index-2.pinecone.io",
                status=IndexModelStatus(ready=True, state="Ready"),
                deletion_protection="disabled",
                spec={
                    "pod": {
                        "environment": "us-west1-gcp",
                        "pod_type": "p1.x1",
                        "pods": 1,
                        "replicas": 1,
                        "shards": 1,
                    }
                },
            ).index,
        ],
        _check_type=False,
    )


class TestIndexList:
    def test_index_list_has_length(self, index_list_response):
        assert len(IndexList(index_list_response)) == 2

    def test_index_list_is(self, index_list_response):
        iil = IndexList(index_list_response)
        assert [i["name"] for i in iil] == ["test-index-1", "test-index-2"]
        assert [i["dimension"] for i in iil] == [2, 3]
        assert [i["metric"] for i in iil] == ["cosine", "cosine"]

    def test_index_list_names_syntactic_sugar(self, index_list_response):
        iil = IndexList(index_list_response)
        assert iil.names() == ["test-index-1", "test-index-2"]

    def test_index_list_getitem(self, index_list_response):
        iil = IndexList(index_list_response)
        input = index_list_response
        assert input.indexes[0].name == iil[0].name
        # Test wrapped IndexModel compatibility properties
        assert iil[0].dimension == 2
        assert iil[0].metric == "cosine"
        assert input.indexes[0].host == iil[0].host
        assert input.indexes[0].deletion_protection == iil[0].deletion_protection
        assert iil[0].deletion_protection == "enabled"

        assert input.indexes[1].name == iil[1].name

    def test_index_list_proxies_methods(self, index_list_response):
        # Forward compatibility, in case we add more attributes to IndexList for pagination
        assert IndexList(index_list_response).indexes[0].name == index_list_response.indexes[0].name

    def test_when_results_are_empty(self):
        iil = IndexList(OpenApiIndexList(indexes=[]))
        assert len(iil) == 0
        assert iil.index_list.indexes == []
        assert iil.indexes == []
        assert iil.names() == []
