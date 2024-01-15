import os
import pytest
import random
import time
from pinecone import Pinecone, NotFoundException, PineconeApiException
from ..helpers import generate_index_name, get_environment_var

@pytest.fixture()
def client():
    api_key = get_environment_var('PINECONE_API_KEY')
    return Pinecone(api_key=api_key)

@pytest.fixture()
def pod_environment():
    return get_environment_var('POD_ENVIRONMENT', 'us-east1-gcp')

@pytest.fixture()
def serverless_cloud():
    return get_environment_var('SERVERLESS_CLOUD', 'aws')

@pytest.fixture()
def serverless_region():
    return get_environment_var('SERVERLESS_REGION', 'us-west-2')

@pytest.fixture()
def environment():
    return get_environment_var('PINECONE_ENVIRONMENT')

@pytest.fixture()
def create_sl_index_params(index_name, serverless_cloud, serverless_region):
    spec = {"serverless": {
        'cloud': serverless_cloud,
        'region': serverless_region
    }}
    return dict(name=index_name, dimension=10, metric='cosine', spec=spec)

@pytest.fixture()
def create_pod_index_params(index_name, pod_environment):
    spec = { 
        'pod': {
            'environment': pod_environment,
            'pod_type': 'p1.x1'
        }
    }
    return dict(name=index_name, dimension=10, metric='cosine', spec=spec, timeout=-1)

@pytest.fixture()
def random_vector():
    return [random.uniform(0, 1) for _ in range(10)]

@pytest.fixture()
def index_name(request):
    test_name = request.node.name
    return generate_index_name(test_name)

@pytest.fixture()
def ready_sl_index(client, index_name, create_sl_index_params):
    create_sl_index_params['timeout'] = None
    client.create_index(**create_sl_index_params)
    yield index_name
    client.delete_index(index_name, -1)

@pytest.fixture()
def notready_sl_index(client, index_name, create_sl_index_params):
    client.create_index(**create_sl_index_params, timeout=-1)
    yield index_name

@pytest.fixture()
def ready_pod_index(client, index_name, create_pod_index_params):
    del create_pod_index_params['timeout']
    client.create_index(**create_pod_index_params)
    yield index_name

@pytest.fixture()
def notready_pod_index(client, index_name, create_pod_index_params):
    client.create_index(**create_pod_index_params)
    yield index_name

def delete_with_retry(client, index_name, retries=0, sleep_interval=5):
    print('Deleting index ' + index_name + ', retry ' + str(retries) + ', next sleep interval ' + str(sleep_interval))
    try:
        client.delete_index(index_name, -1)
    except NotFoundException:
        pass
    except PineconeApiException as e:
        if e.error.code == 'PRECONDITON_FAILED':
            if retries > 5:
                raise 'Unable to delete index ' + index_name
            time.sleep(sleep_interval)
            delete_with_retry(client, index_name, retries + 1, sleep_interval * 2)
        else:
            print(e.__class__)
            print(e)
            raise 'Unable to delete index ' + index_name
    except Exception as e:
        print(e.__class__)
        print(e)
        raise 'Unable to delete index ' + index_name

@pytest.fixture(autouse=True)
def cleanup(client, index_name):
    yield

    try:
       client.delete_index(index_name, -1)
    except:
       pass

@pytest.fixture(autouse=True, scope='session')
def cleanup_all():
    yield

    client = Pinecone()
    for index in client.list_indexes():
        buildNumber = os.getenv('GITHUB_BUILD_NUMBER')
        if index.name.startswith(buildNumber):
            try:
                delete_with_retry(client, index.name)
            except:
                pass