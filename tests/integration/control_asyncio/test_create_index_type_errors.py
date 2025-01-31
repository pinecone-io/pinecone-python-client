import pytest
from pinecone import PineconeApiException, PineconeApiTypeError, PineconeAsyncio


@pytest.mark.asyncio
class TestCreateIndexTypeErrorCases:
    async def test_create_index_with_invalid_str_dimension(
        self, api_key_fixture, create_sl_index_params
    ):
        pc = PineconeAsyncio(api_key=api_key_fixture)

        create_sl_index_params["dimension"] = "10"
        with pytest.raises(PineconeApiTypeError):
            await pc.create_index(**create_sl_index_params)

    async def test_create_index_with_missing_dimension(
        self, api_key_fixture, create_sl_index_params
    ):
        pc = PineconeAsyncio(api_key=api_key_fixture)

        del create_sl_index_params["dimension"]
        with pytest.raises(PineconeApiException):
            await pc.create_index(**create_sl_index_params)
