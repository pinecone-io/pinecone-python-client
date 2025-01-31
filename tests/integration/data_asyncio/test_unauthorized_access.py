import pytest
from pinecone import PineconeAsyncio, ForbiddenException


@pytest.mark.asyncio
async def test_unauthorized_requests_rejected(index_host):
    pc = PineconeAsyncio(api_key="invalid_key")
    asyncio_idx = pc.Index(host=index_host)

    with pytest.raises(ForbiddenException) as e:
        await asyncio_idx.describe_index_stats()
    assert "Wrong API key" in str(e.value)
