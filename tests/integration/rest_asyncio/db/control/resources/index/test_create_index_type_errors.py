import pytest
from pinecone import PineconeApiException, PineconeApiTypeError, PineconeAsyncio


@pytest.mark.asyncio
class TestCreateIndexTypeErrorCases:
    @pytest.mark.skip(
        reason="Covered by unit tests in tests/unit/openapi_support/test_endpoint_validation.py"
    )
    async def test_create_index_with_invalid_str_dimension(self, create_sl_index_params):
        pc = PineconeAsyncio()

        create_sl_index_params["dimension"] = "10"
        with pytest.raises(PineconeApiTypeError):
            await pc.create_index(**create_sl_index_params)
        await pc.close()

    @pytest.mark.skip(
        reason="Covered by unit tests in tests/unit/openapi_support/test_endpoint_validation.py"
    )
    async def test_create_index_with_missing_dimension(self, create_sl_index_params):
        pc = PineconeAsyncio()

        del create_sl_index_params["dimension"]
        with pytest.raises(PineconeApiException):
            await pc.create_index(**create_sl_index_params)
        await pc.close()
