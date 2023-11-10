import pytest

class TestCreateIndexHappyPath:
    def test_create_index(self, client, create_index_params):        
        name = create_index_params['name']
        dimension = create_index_params['dimension']
        client.create_index(**create_index_params)
        desc = client.describe_index(name)
        assert desc.name == name
        assert desc.dimension == dimension
        assert desc.metric == 'cosine'

    @pytest.mark.parametrize('metric', ['cosine', 'euclidean', 'dotproduct'])
    def test_create_index_with_metric(self, client, create_index_params, metric):
        create_index_params['metric'] = metric
        client.create_index(**create_index_params)
        desc = client.describe_index(create_index_params['name'])
        assert desc.metric == metric

    @pytest.mark.skip(reason='Bug filed https://app.asana.com/0/1205078872348810/1205917627868150')
    def test_create_index_w_metadata_config(self, client, create_index_params):
        create_index_params['metadata_config'] = {'indexed': ['genre', 'rating']}
        client.create_index(**create_index_params)
        desc = client.describe_index(create_index_params['name'])
        assert desc.metadata_config == {'indexed': 'genre'}
