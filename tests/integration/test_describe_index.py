import pytest
from pinecone import IndexDescription

class TestDescribeIndex:
    def test_describe_index_when_ready(self, client, ready_index, create_index_params):
        description = client.describe_index(ready_index)
        assert type(description) == IndexDescription
        assert description.name == ready_index
        assert description.dimension == create_index_params['dimension']
        assert description.metric == create_index_params['metric']
        assert description.capacity_mode == create_index_params['capacity_mode'] 

        assert type(description.status.host) == str
        assert description.status.host != ""
        assert description.status.state == 'Ready'
        assert ready_index in description.status.host
        assert description.status.ready == True

    def test_describe_index_when_not_ready(self, client, notready_index, create_index_params):
        description = client.describe_index(notready_index)

        assert type(description) == IndexDescription
        assert description.name == notready_index
        assert description.dimension == create_index_params['dimension']
        assert description.metric == create_index_params['metric']
        assert description.capacity_mode == create_index_params['capacity_mode'] 

        assert type(description.status.host) == str
        assert description.status.host != ""
        assert notready_index in description.status.host
        assert description.status.ready == False