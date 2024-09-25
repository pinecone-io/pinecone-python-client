import pytest
from pinecone import IndexList
from pinecone.core.openapi.control.models import (
    IndexList as OpenApiIndexList,
    IndexModel as OpenApiIndexModel,
    IndexModelSpec,
    IndexModelStatus,
    DeletionProtection,
    PodSpec as OpenApiPodSpec,
)


@pytest.fixture
def index_list_response():
    return OpenApiIndexList(
        indexes=[
            OpenApiIndexModel(
                name="test-index-1",
                dimension=2,
                metric="cosine",
                host="https://test-index-1.pinecone.io",
                status=IndexModelStatus(ready=True, state="Ready"),
                deletion_protection=DeletionProtection("enabled"),
                spec=IndexModelSpec(
                    pod=OpenApiPodSpec(
                        environment="us-west1-gcp", pod_type="p1.x1", pods=1, replicas=1, shards=1
                    )
                ),
            ),
            OpenApiIndexModel(
                name="test-index-2",
                dimension=3,
                metric="cosine",
                host="https://test-index-2.pinecone.io",
                status=IndexModelStatus(ready=True, state="Ready"),
                deletion_protection=DeletionProtection("disabled"),
                spec=IndexModelSpec(
                    pod=OpenApiPodSpec(
                        environment="us-west1-gcp", pod_type="p1.x1", pods=1, replicas=1, shards=1
                    )
                ),
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
        assert [i["dimension"] for i in iil] == [2, 3]
        assert [i["metric"] for i in iil] == ["cosine", "cosine"]

    def test_index_list_names_syntactic_sugar(self, index_list_response):
        iil = IndexList(index_list_response)
        assert iil.names() == ["test-index-1", "test-index-2"]

    def test_index_list_getitem(self, index_list_response):
        iil = IndexList(index_list_response)
        input = index_list_response
        assert input.indexes[0].name == iil[0].name
        assert input.indexes[0].dimension == iil[0].dimension
        assert input.indexes[0].metric == iil[0].metric
        assert input.indexes[0].host == iil[0].host
        assert input.indexes[0].deletion_protection.value == iil[0].deletion_protection
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
