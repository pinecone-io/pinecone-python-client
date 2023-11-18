import pytest
from pinecone import IndexDescription

class TestDescribeIndex:
    def test_describe_index_when_ready(self, client, ready_sl_index, create_sl_index_params):
        description = client.describe_index(ready_sl_index)
        assert type(description) == IndexDescription
        assert description.name == ready_sl_index
        assert description.dimension == create_sl_index_params['dimension']
        assert description.metric == create_sl_index_params['metric']
        assert description.capacity_mode == create_sl_index_params['capacity_mode'] 

        assert type(description.status.host) == str
        assert description.status.host != ""
        assert description.status.state == 'Ready'
        assert ready_sl_index in description.status.host
        assert description.status.ready == True

    def test_describe_index_when_not_ready(self, client, notready_sl_index, create_sl_index_params):
        description = client.describe_index(notready_sl_index)

        assert type(description) == IndexDescription
        assert description.name == notready_sl_index
        assert description.dimension == create_sl_index_params['dimension']
        assert description.metric == create_sl_index_params['metric']

        assert type(description.status.host) == str
        assert description.status.host != ""
        assert notready_sl_index in description.status.host
        assert description.status.ready == False