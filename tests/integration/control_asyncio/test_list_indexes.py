import pytest
from pinecone import IndexModel, PineconeAsyncio


@pytest.mark.asyncio
class TestListIndexes:
    async def test_list_indexes_includes_ready_indexes(
        self, api_key_fixture, ready_sl_index, create_sl_index_params
    ):
        pc = PineconeAsyncio(api_key=api_key_fixture)

        list_response = await pc.list_indexes()
        assert len(list_response.indexes) != 0
        assert isinstance(list_response.indexes[0], IndexModel)

        created_index = [index for index in list_response.indexes if index.name == ready_sl_index][
            0
        ]
        assert created_index.name == ready_sl_index
        assert created_index.dimension == create_sl_index_params["dimension"]
        assert created_index.metric == create_sl_index_params["metric"]
        assert ready_sl_index in created_index.host

    async def test_list_indexes_includes_not_ready_indexes(
        self, api_key_fixture, notready_sl_index
    ):
        pc = PineconeAsyncio(api_key=api_key_fixture)

        list_response = await pc.list_indexes()
        assert len(list_response.indexes) != 0
        assert isinstance(list_response.indexes[0], IndexModel)

        created_index = [
            index for index in list_response.indexes if index.name == notready_sl_index
        ][0]
        assert created_index.name == notready_sl_index
        assert notready_sl_index in created_index.name
