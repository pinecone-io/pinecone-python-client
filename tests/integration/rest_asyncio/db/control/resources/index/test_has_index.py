import pytest
from tests.integration.helpers import random_string
from pinecone import PineconeAsyncio


@pytest.mark.asyncio
class TestHasIndex:
    async def test_index_exists_success(self, create_sl_index_params):
        pc = PineconeAsyncio()

        name = create_sl_index_params["name"]
        await pc.create_index(**create_sl_index_params)
        has_index = await pc.has_index(name)
        assert has_index == True
        await pc.close()

    async def test_index_does_not_exist(self):
        pc = PineconeAsyncio()

        name = random_string(8)
        has_index = await pc.has_index(name)
        assert has_index == False
        await pc.close()

    async def test_has_index_with_null_index_name(self):
        pc = PineconeAsyncio()

        has_index = await pc.has_index("")
        assert has_index == False
        await pc.close()
