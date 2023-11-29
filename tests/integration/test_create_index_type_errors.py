import pytest
from pinecone import PineconeApiTypeError

class TestCreateIndexTypeErrorCases:
    def test_create_index_with_invalid_str_dimension(self, client, create_sl_index_params):
        create_sl_index_params['dimension'] = '10'
        with pytest.raises(PineconeApiTypeError):
            client.create_index(**create_sl_index_params)

    def test_create_index_with_missing_dimension(self, client, create_sl_index_params):
        del create_sl_index_params['dimension']
        with pytest.raises(TypeError):
            client.create_index(**create_sl_index_params)