import pytest
from pinecone import Pinecone
from .helpers.helpers import generate_index_name, get_environment_var

@pytest.fixture()
def client():
    api_key = get_environment_var('PINECONE_API_KEY')
    return Pinecone(api_key=api_key)

@pytest.fixture()
def capacity_mode1():
    return get_environment_var('TEST_CAPACITY_MODE_1')

@pytest.fixture()
def capacity_mode2():
    return get_environment_var('TEST_CAPACITY_MODE_2')

@pytest.fixture()
def create_index_params(index_name, capacity_mode1):
    return dict(name=index_name, dimension=10, metric='cosine', cloud='aws', region='us-east1', capacity_mode=capacity_mode1, timeout=-1)

@pytest.fixture()
def index_name(request):
    test_name = request.node.name
    return generate_index_name(test_name)

@pytest.fixture()
def ready_index(client, index_name, create_index_params):
    del create_index_params['timeout']
    client.create_index(**create_index_params)
    yield index_name
    client.delete_index(index_name, -1)

@pytest.fixture()
def notready_index(client, index_name, create_index_params):
    client.create_index(**create_index_params)
    yield index_name
    client.delete_index(index_name, -1)

@pytest.fixture(autouse=True)
def cleanup(client, index_name):
    yield

    client.delete_index(index_name, -1)

    try:
       client.delete_index(index_name, -1)
    except:
       pass

@pytest.fixture(autouse=True, scope='session')
def cleanup_all():
    yield

    client = Pinecone()
    for index in client.list_indexes().databases:
        try:
            client.delete_index(index.name, -1)
        except:
            pass