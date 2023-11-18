from pinecone import IndexModel

class TestListIndexes():
    def test_list_indexes_includes_ready_indexes(self, client, ready_index, create_index_params):
        list_response = client.list_indexes()
        assert len(list_response.indexes) != 0
        assert type(list_response.indexes[0]) == IndexModel
        
        created_index = [index for index in list_response.indexes if index.name == ready_index][0]
        assert created_index.name == ready_index
        assert created_index.dimension == create_index_params['dimension']
        assert created_index.metric == create_index_params['metric']
        assert ready_index in created_index.host
        assert created_index.capacity_mode == create_index_params['capacity_mode']

    def test_list_indexes_includes_not_ready_indexes(self, client, notready_index):
        list_response = client.list_indexes()
        assert len(list_response.indexes) != 0
        assert type(list_response.indexes[0]) == IndexModel

        created_index = [index for index in list_response.indexes if index.name == notready_index][0]
        assert created_index.name == notready_index
        assert notready_index in created_index.name
