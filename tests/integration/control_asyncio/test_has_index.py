import pytest
from tests.integration.helpers import random_string
from pinecone import PineconeAsyncio


@pytest.mark.asyncio
class TestHasIndex:
    async def test_index_exists_success(self, api_key_fixture, create_sl_index_params):
        pc = PineconeAsyncio(api_key=api_key_fixture)

        name = create_sl_index_params["name"]
        await pc.create_index(**create_sl_index_params)
        has_index = await pc.has_index(name)
        assert has_index == True

    async def test_index_does_not_exist(self, api_key_fixture):
        pc = PineconeAsyncio(api_key=api_key_fixture)

        name = random_string(8)
        has_index = await pc.has_index(name)
        assert has_index == False

    async def test_has_index_with_null_index_name(self, api_key_fixture):
        pc = PineconeAsyncio(api_key=api_key_fixture)

        has_index = await pc.has_index("")
        assert has_index == False
