import pytest

class TestCreateSLIndexHappyPath:
    def test_create_index(self, client, create_sl_index_params):        
        name = create_sl_index_params['name']
        dimension = create_sl_index_params['dimension']
        client.create_index(**create_sl_index_params)
        desc = client.describe_index(name)
        assert desc.name == name
        assert desc.dimension == dimension
        assert desc.metric == 'cosine'

    @pytest.mark.parametrize('metric', ['cosine', 'euclidean', 'dotproduct'])
    def test_create_index_with_metric(self, client, create_sl_index_params, metric):
        create_sl_index_params['metric'] = metric
        client.create_index(**create_sl_index_params)
        desc = client.describe_index(create_sl_index_params['name'])
        assert desc.metric == metric