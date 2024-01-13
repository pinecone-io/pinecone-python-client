import pytest
from pinecone import CollectionDescription

@pytest.mark.skip()
class TestCollection:
    def test_create_collection_from_ready_index_w_data(self, client, ready_index):
        index = client.Index(ready_index)
        index.upsert([{'id': '1', 'values': [0.1, 0.2]}])

        name = 'test_collection'
        client.create_collection(name, ready_index)
        desc = client.describe_collection(name)
        assert type(desc) == CollectionDescription
        assert desc.name == name
        assert desc.index == ready_index

        client.delete_collection(name)

    def test_create_collection_from_ready_index_w_no_data(self, client, ready_index):
        name = 'test_collection'
        client.create_collection(name, ready_index)
        desc = client.describe_collection(name)
        assert type(desc) == CollectionDescription
        assert desc.name == name
        assert desc.index == ready_index

        client.delete_collection(name)

    def test_create_collection_from_not_ready_index(self, client, notready_index):
        name = 'test_collection'
        with pytest.raises(Exception) as e:
            client.create_collection(name, notready_index)
        assert 'Index is not ready' in str(e.value)

    def test_create_collection_with_invalid_index(self, client):
        name = 'test_collection'
        with pytest.raises(Exception) as e:
            client.create_collection(name, 'invalid_index')
        assert 'Index not found' in str(e.value)