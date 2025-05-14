from pinecone import IndexModel


class TestListIndexes:
    def test_list_indexes_includes_ready_indexes(self, pc, ready_sl_index, create_index_params):
        list_response = pc.db.index.list()
        assert len(list_response.indexes) != 0
        assert isinstance(list_response.indexes[0], IndexModel)

        created_index = [index for index in list_response.indexes if index.name == ready_sl_index][
            0
        ]
        assert created_index.name == ready_sl_index
        assert created_index.dimension == create_index_params["dimension"]
        assert created_index.metric == create_index_params["metric"]
        assert ready_sl_index in created_index.host

    def test_list_indexes_includes_not_ready_indexes(self, pc, notready_sl_index):
        list_response = pc.db.index.list()
        assert len(list_response.indexes) != 0
        assert isinstance(list_response.indexes[0], IndexModel)

        created_index = [
            index for index in list_response.indexes if index.name == notready_sl_index
        ][0]
        assert created_index.name == notready_sl_index
        assert notready_sl_index in created_index.name
