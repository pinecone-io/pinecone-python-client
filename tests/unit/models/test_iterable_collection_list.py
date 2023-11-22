import pytest
from pinecone import IterableCollectionList
from pinecone.core.client.models import CollectionList, CollectionModel

@pytest.fixture
def collection_list_response():
    return CollectionList(
            collections=[
               CollectionModel(name='collection1', size=10000, status='Ready', dimension=1536, record_count=1000),
               CollectionModel(name='collection2', size=20000, status='Ready', dimension=256, record_count=2000),
            ],
        )

class TestIterableCollectionList:
    def test_iterable_collection_list_has_length(self, collection_list_response):
        assert len(IterableCollectionList(collection_list_response)) == 2

    def test_iterable_collection_list_is_iterable(self, collection_list_response):
        icl = IterableCollectionList(collection_list_response)
        assert [i['name'] for i in icl] == ['collection1', 'collection2']
        assert [i.record_count for i in icl] == [1000, 2000]

    def test_iterable_collection_list_getitem(self, collection_list_response):
        icl = IterableCollectionList(collection_list_response)
        assert icl[0] == collection_list_response.collections[0]
        assert icl[1] == collection_list_response.collections[1]

    def test_iterable_collection_list_proxies_methods(self, collection_list_response):
        # Forward compatibility, in case we add more attributes to IndexList for pagination
        assert IterableCollectionList(collection_list_response).collection_list.collections == collection_list_response.collections