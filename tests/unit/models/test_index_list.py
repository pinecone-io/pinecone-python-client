import pytest
from pinecone import IndexList
from pinecone.core.client.models import IndexList as OpenApiIndexList

@pytest.fixture
def index_list_response():
    return OpenApiIndexList(
            indexes=[
                {
                    "name": "test-index-1",
                    "dimension": 2,
                    "metric": "cosine",
                    "spec": {
                        "pod": {
                            "environment": "us-west1-gcp",
                            "pod_type": "p1.x1",
                            "pods": 1,
                            "replicas": 1,
                            "shards": 1
                        }
                    }
                },
                {
                    "name": "test-index-2",
                    "dimension": 3,
                    "metric": "cosine",
                    "spec": {
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-west-2"
                        }
                    }
                },
            ], _check_type=False
        )

class TestIndexList:
    def test_index_list_has_length(self, index_list_response):
        assert len(IndexList(index_list_response)) == 2

    def test_index_list_is(self, index_list_response):
        iil = IndexList(index_list_response)
        assert [i['name'] for i in iil] == ['test-index-1', 'test-index-2']
        assert [i['dimension'] for i in iil] == [2, 3]
        assert [i['metric'] for i in iil] == ['cosine', 'cosine']

    def test_index_list_names_syntactic_sugar(self, index_list_response):
        iil = IndexList(index_list_response)
        assert iil.names() == ['test-index-1', 'test-index-2']

    def test_index_list_getitem(self, index_list_response):
        iil = IndexList(index_list_response)
        assert iil[0] == index_list_response.indexes[0]
        assert iil[1] == index_list_response.indexes[1]

    def test_index_list_proxies_methods(self, index_list_response):
        # Forward compatibility, in case we add more attributes to IndexList for pagination
        assert IndexList(index_list_response).index_list.indexes == index_list_response.indexes

    def test_when_results_are_empty(self):
        iil = IndexList(OpenApiIndexList(indexes=[]))
        assert len(iil) == 0
        assert iil.index_list.indexes == []
        assert iil.indexes == []
        assert iil.names() == []

