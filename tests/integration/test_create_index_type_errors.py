import pytest
from pinecone import ApiTypeError

@pytest.fixture()
def create_index_params(index_name, capacity_mode1):
    return dict(name=index_name, dimension=10, cloud='aws', region='us-east1', capacity_mode=capacity_mode1, timeout=-1)

class TestCreateIndexTypeErrorCases:
    def test_create_index_with_invalid_str_dimension(self, client, create_index_params):
        create_index_params['dimension'] = '10'
        with pytest.raises(ApiTypeError):
            client.create_index(**create_index_params)

    def test_create_index_with_missing_dimension(self, client, create_index_params):
        del create_index_params['dimension']
        with pytest.raises(TypeError):
            client.create_index(**create_index_params)