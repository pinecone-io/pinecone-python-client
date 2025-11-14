import pytest
from pinecone import PineconeAsyncio, ForbiddenException


@pytest.mark.asyncio
async def test_unauthorized_requests_rejected(index_host):
    async with PineconeAsyncio(api_key="invalid_key") as pc:
        async with pc.IndexAsyncio(host=index_host) as asyncio_idx:
            with pytest.raises(ForbiddenException) as e:
                await asyncio_idx.describe_index_stats()
            assert "Wrong API key" in str(e.value)
