import pytest 

class TestCreateIndexWithTimeout:
    def test_create_index_default_timeout(self, client, create_sl_index_params):
        create_sl_index_params['timeout'] = None
        client.create_index(**create_sl_index_params)
        # Waits infinitely for index to be ready
        desc = client.describe_index(create_sl_index_params['name'])
        assert desc.status.ready == True

    def test_create_index_when_timeout_set(self, client, create_sl_index_params):
        create_sl_index_params['timeout'] = 1000 # effectively infinite, but different code path from None
        client.create_index(**create_sl_index_params)
        desc = client.describe_index(create_sl_index_params['name'])
        assert desc.status.ready == True

    def test_create_index_when_timeout_error(self, client, create_sl_index_params):
        create_sl_index_params['timeout'] = 1
        with pytest.raises(TimeoutError):
            client.create_index(**create_sl_index_params)

    def test_create_index_with_negative_timeout(self, client, create_sl_index_params):
        create_sl_index_params['timeout'] = -1
        client.create_index(**create_sl_index_params)
        desc = client.describe_index(create_sl_index_params['name'])
        # Returns immediately without waiting for index to be ready
        assert desc.status.ready == False

