import pytest
from pinecone import PineconeApiException, PineconeApiValueError

class TestCreateIndexApiErrorCases:
    def test_create_index_with_invalid_name(self, client, create_sl_index_params):
        create_sl_index_params['name'] = '-invalid-name'
        with pytest.raises(PineconeApiException):
            client.create_index(**create_sl_index_params)

    def test_create_index_invalid_metric(self, client, create_sl_index_params):
        create_sl_index_params['metric'] = 'invalid'
        with pytest.raises(PineconeApiValueError):
            client.create_index(**create_sl_index_params)

    def test_create_index_with_invalid_neg_dimension(self, client, create_sl_index_params):
        create_sl_index_params['dimension'] = -1
        with pytest.raises(PineconeApiValueError):
            client.create_index(**create_sl_index_params)

    def test_create_index_that_already_exists(self, client, create_sl_index_params):
        client.create_index(**create_sl_index_params)
        with pytest.raises(PineconeApiException):
            client.create_index(**create_sl_index_params)

    @pytest.mark.skip(reason='Bug filed https://app.asana.com/0/1205078872348810/1205917627868143')
    def test_create_index_w_incompatible_options(self, client, create_sl_index_params):
        create_sl_index_params['pod_type'] = 'p1.x2'
        create_sl_index_params['environment'] = 'us-east1-gcp'
        create_sl_index_params['replicas'] = 2
        with pytest.raises(PineconeApiException):
            client.create_index(**create_sl_index_params)