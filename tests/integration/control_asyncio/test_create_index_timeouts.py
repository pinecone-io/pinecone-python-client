from pinecone import PineconeAsyncio
import pytest


@pytest.mark.asyncio
class TestCreateIndexWithTimeout:
    async def test_create_index_default_timeout(self, create_sl_index_params):
        pc = PineconeAsyncio()

        create_sl_index_params["timeout"] = None
        await pc.create_index(**create_sl_index_params)
        # Waits infinitely for index to be ready
        desc = await pc.describe_index(create_sl_index_params["name"])
        assert desc.status.ready == True

    async def test_create_index_when_timeout_set(self, create_sl_index_params):
        pc = PineconeAsyncio()

        create_sl_index_params["timeout"] = (
            1000  # effectively infinite, but different code path from None
        )
        await pc.create_index(**create_sl_index_params)
        desc = await pc.describe_index(create_sl_index_params["name"])
        assert desc.status.ready == True

    async def test_create_index_with_negative_timeout(self, create_sl_index_params):
        pc = PineconeAsyncio()

        create_sl_index_params["timeout"] = -1
        await pc.create_index(**create_sl_index_params)
        desc = await pc.describe_index(create_sl_index_params["name"])
        # Returns immediately without waiting for index to be ready
        assert desc.status.ready in [False, True]
