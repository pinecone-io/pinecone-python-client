import pytest
from pinecone import IndexModel

class TestDescribeIndex:
    def test_describe_index_when_ready(self, client, ready_sl_index, create_sl_index_params):
        description = client.describe_index(ready_sl_index)
        
        assert type(description) == IndexModel
        assert description.name == ready_sl_index
        assert description.dimension == create_sl_index_params['dimension']
        assert description.metric == create_sl_index_params['metric']
        assert description.spec.serverless['cloud'] == create_sl_index_params['spec']['serverless']['cloud']
        assert description.spec.serverless['region'] == create_sl_index_params['spec']['serverless']['region']

        assert type(description.host) == str
        assert description.host != ""
        assert ready_sl_index in description.host

        assert description.status.state == 'Ready'
        assert description.status.ready == True

    def test_describe_index_when_not_ready(self, client, notready_sl_index, create_sl_index_params):
        description = client.describe_index(notready_sl_index)

        assert type(description) == IndexModel
        assert description.name == notready_sl_index
        assert description.dimension == create_sl_index_params['dimension']
        assert description.metric == create_sl_index_params['metric']
        assert description.spec.serverless['cloud'] == create_sl_index_params['spec']['serverless']['cloud']
        assert description.spec.serverless['region'] == create_sl_index_params['spec']['serverless']['region']

        assert type(description.host) == str
        assert description.host != ""
        assert notready_sl_index in description.host

        assert description.status.ready == False
        assert description.status.state in ['Ready', 'Initializing']