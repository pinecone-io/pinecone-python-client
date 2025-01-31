import pytest
from pinecone import IndexModel, PineconeAsyncio


@pytest.mark.asyncio
class TestDescribeIndex:
    async def test_describe_index_when_ready(
        self, api_key_fixture, ready_sl_index, create_sl_index_params
    ):
        pc = PineconeAsyncio(api_key=api_key_fixture)
        description = await pc.describe_index(ready_sl_index)

        assert isinstance(description, IndexModel)
        assert description.name == ready_sl_index
        assert description.dimension == create_sl_index_params["dimension"]
        assert description.metric == create_sl_index_params["metric"]
        assert (
            description.spec.serverless["cloud"]
            == create_sl_index_params["spec"]["serverless"]["cloud"]
        )
        assert (
            description.spec.serverless["region"]
            == create_sl_index_params["spec"]["serverless"]["region"]
        )

        assert isinstance(description.host, str)
        assert description.host != ""
        assert ready_sl_index in description.host

        assert description.status.state == "Ready"
        assert description.status.ready == True

    async def test_describe_index_when_not_ready(
        self, api_key_fixture, notready_sl_index, create_sl_index_params
    ):
        pc = PineconeAsyncio(api_key=api_key_fixture)
        description = await pc.describe_index(notready_sl_index)

        assert isinstance(description, IndexModel)
        assert description.name == notready_sl_index
        assert description.dimension == create_sl_index_params["dimension"]
        assert description.metric == create_sl_index_params["metric"]
        assert (
            description.spec.serverless["cloud"]
            == create_sl_index_params["spec"]["serverless"]["cloud"]
        )
        assert (
            description.spec.serverless["region"]
            == create_sl_index_params["spec"]["serverless"]["region"]
        )

        assert isinstance(description.host, str)
        assert description.host != ""
        assert notready_sl_index in description.host
