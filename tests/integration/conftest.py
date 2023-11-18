import pytest
from pinecone import Pinecone
from .helpers.helpers import generate_index_name, get_environment_var

@pytest.fixture()
def client():
    api_key = get_environment_var('PINECONE_API_KEY')
    return Pinecone(api_key=api_key)

@pytest.fixture()
def environment():
    return get_environment_var('PINECONE_ENVIRONMENT')

@pytest.fixture()
def create_sl_index_params(index_name):
    spec = {
        'cloud': 'aws',
        'region': 'us-east1'
    }
    return dict(name=index_name, dimension=10, metric='cosine', spec=spec, timeout=-1)

@pytest.fixture()
def create_pod_index_params(index_name, environment):
    spec = {
        'environment': environment
    }
    return dict(name=index_name, dimension=10, metric='cosine', spec=spec, timeout=-1)

@pytest.fixture()
def index_name(request):
    test_name = request.node.name
    return generate_index_name(test_name)

@pytest.fixture()
def ready_sl_index(client, index_name, create_sl_index_params):
    del create_sl_index_params['timeout']
    client.create_index(**create_sl_index_params)
    yield index_name
    client.delete_index(index_name, -1)

@pytest.fixture()
def notready_sl_index(client, index_name, create_sl_index_params):
    client.create_index(**create_sl_index_params)
    yield index_name
    client.delete_index(index_name, -1)

@pytest.fixture()
def ready_pod_index(client, index_name, create_pod_index_params):
    del create_pod_index_params['timeout']
    client.create_index(**create_pod_index_params)
    yield index_name
    client.delete_index(index_name, -1)

@pytest.fixture()
def notready_pod_index(client, index_name, create_pod_index_params):
    client.create_index(**create_pod_index_params)
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