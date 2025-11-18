import pytest
from pinecone import PineconeApiException, PineconeApiValueError, PineconeAsyncio


@pytest.mark.asyncio
class TestCreateIndexApiErrorCases:
    async def test_create_index_with_invalid_name(self, create_sl_index_params):
        pc = PineconeAsyncio()
        create_sl_index_params["name"] = "Invalid-name"
        with pytest.raises(PineconeApiException):
            await pc.create_index(**create_sl_index_params)
        await pc.close()

    async def test_create_index_invalid_metric(self, create_sl_index_params):
        pc = PineconeAsyncio()
        create_sl_index_params["metric"] = "invalid"
        with pytest.raises(PineconeApiException):
            await pc.create_index(**create_sl_index_params)
        await pc.close()

    async def test_create_index_with_invalid_neg_dimension(self, create_sl_index_params):
        pc = PineconeAsyncio()
        create_sl_index_params["dimension"] = -1
        with pytest.raises(PineconeApiValueError):
            await pc.create_index(**create_sl_index_params)
        await pc.close()

    async def test_create_index_that_already_exists(self, create_sl_index_params):
        async with PineconeAsyncio() as pc:
            await pc.create_index(**create_sl_index_params)
            with pytest.raises(PineconeApiException):
                await pc.create_index(**create_sl_index_params)

    @pytest.mark.skip(reason="Bug filed https://app.asana.com/0/1205078872348810/1205917627868143")
    async def test_create_index_w_incompatible_options(self, create_sl_index_params):
        async with PineconeAsyncio() as pc:
            create_sl_index_params["pod_type"] = "p1.x2"
            create_sl_index_params["environment"] = "us-east1-gcp"
            create_sl_index_params["replicas"] = 2
            with pytest.raises(PineconeApiException):
                await pc.create_index(**create_sl_index_params)
