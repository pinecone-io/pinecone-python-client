import pytest
from pinecone import IterableIndexList
from pinecone.core.client.models import IndexList

@pytest.fixture
def index_list_response():
    return IndexList(
            indexes=[
                {
                    "name": "test-index-1",
                    "dimension": 2,
                    "metric": "cosine",
                    "capacity_mode": "serverless",
                    "status": {
                        "host": "test-index-1-host",
                        "state": "Ready",
                        "ready": True,
                    },
                },
                {
                    "name": "test-index-2",
                    "dimension": 2,
                    "metric": "cosine",
                    "capacity_mode": "serverless",
                    "status": {
                        "host": "test-index-2-host",
                        "state": "Ready",
                        "ready": True,
                    },
                },
            ], _check_type=False
        )

class TestIterableIndexList:
    def test_iterable_index_list_has_length(self, index_list_response):
        assert len(IterableIndexList(index_list_response)) == 2

    def test_iterable_index_list_is_iterable(self, index_list_response):
        iil = IterableIndexList(index_list_response)
        assert [i['name'] for i in iil] == ['test-index-1', 'test-index-2']

    def test_iterable_index_list_getitem(self, index_list_response):
        iil = IterableIndexList(index_list_response)
        assert iil[0] == index_list_response.indexes[0]
        assert iil[1] == index_list_response.indexes[1]

    def test_iterable_index_list_proxies_methods(self, index_list_response):
        # Forward compatibility, in case we add more attributes to IndexList for pagination
        assert IterableIndexList(index_list_response).index_list.indexes == index_list_response.indexes


