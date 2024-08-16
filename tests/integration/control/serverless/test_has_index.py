from tests.integration.helpers import random_string


class TestHasIndex:
    def test_index_exists_success(self, client, create_sl_index_params):
        name = create_sl_index_params["name"]
        client.create_index(**create_sl_index_params)
        has_index = client.has_index(name)
        assert has_index == True

    def test_index_does_not_exist(self, client):
        name = random_string(8)
        has_index = client.has_index(name)
        assert has_index == False

    def test_has_index_with_null_index_name(self, client):
        has_index = client.has_index('')
        assert has_index == False