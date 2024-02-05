import pytest
import random
import string
import time
import json
from pinecone import Pinecone, PodSpec
from ...helpers import generate_index_name, get_environment_var

@pytest.fixture()
def client(additional_headers):
    api_key = get_environment_var('PINECONE_API_KEY')
    return Pinecone(
        api_key=api_key,
        additional_headers=additional_headers
    )

@pytest.fixture()
def additional_headers():
    return json.loads(get_environment_var('ADDITIONAL_HEADERS_JSON', '{}'))

@pytest.fixture()
def environment():
    return get_environment_var('PINECONE_ENVIRONMENT')

@pytest.fixture()
def dimension():
    return int(get_environment_var('DIMENSION', 1536))

@pytest.fixture()
def create_index_params(index_name, environment, dimension, metric):
    spec = { 
        'pod': {
            'environment': environment,
            'pod_type': 'p1.x1'
        }
    }
    return dict(
        name=index_name, 
        dimension=dimension, 
        metric=metric, 
        spec=spec, 
        timeout=-1
    )

@pytest.fixture()
def metric():
    return get_environment_var('METRIC', 'cosine')

@pytest.fixture()
def random_vector(dimension):
    def _random_vector():
        return [random.uniform(0, 1) for _ in range(dimension)]
    return _random_vector

@pytest.fixture()
def index_name(request):
    test_name = request.node.name
    return generate_index_name(test_name)

@pytest.fixture()
def ready_index(client, index_name, create_index_params):
    create_index_params['timeout'] = None
    client.create_index(**create_index_params)
    time.sleep(10) # Extra wait, since status is sometimes inaccurate
    yield index_name
    client.delete_index(index_name, -1)

@pytest.fixture()
def notready_index(client, index_name, create_index_params):
    create_index_params.update({'timeout': -1 })
    client.create_index(**create_index_params)
    yield index_name

def index_exists(index_name, client):
    return index_name in client.list_indexes().names()

def random_string():
    return ''.join(random.choice(string.ascii_lowercase) for i in range(10))

@pytest.fixture(scope='session')
def reusabale_index():
    pc = Pinecone(
        api_key=get_environment_var('PINECONE_API_KEY'),
        additional_headers=json.loads(get_environment_var('ADDITIONAL_HEADERS_JSON', '{}'))
    )
    index_name = 'temp-index-' + random_string()
    dimension = int(get_environment_var('DIMENSION', 1536))
    print(f"Creating index {index_name} to prepare a collection...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=get_environment_var('METRIC', 'cosine'),
        spec=PodSpec(
            environment=get_environment_var('PINECONE_ENVIRONMENT'),
        )
    )
    print(f"Created index {index_name}. Waiting 10 seconds to make sure it's ready...")
    time.sleep(10)
    yield index_name
    print(f"Deleting reusable index {index_name}")
    pc.delete_index(index_name)

@pytest.fixture(scope='session')
def reusable_collection(reusabale_index):
    dimension = int(get_environment_var('DIMENSION', 1536))
    pc = Pinecone(
        api_key=get_environment_var('PINECONE_API_KEY'),
        additional_headers=json.loads(get_environment_var('ADDITIONAL_HEADERS_JSON', '{}'))
    )
    
    num_vectors = 10
    vectors = [ 
        (str(i), [random.uniform(0, 1) for _ in range(dimension)]) for i in range(num_vectors) ]
    
    index = pc.Index(reusabale_index)
    index.upsert(vectors=vectors)

    collection_name = 'reused-coll-' + random_string()
    pc.create_collection(
        name=collection_name, 
        source=reusabale_index
    )
    
    time_waited = 0
    desc = pc.describe_collection(collection_name)
    collection_ready = desc['status']
    while collection_ready.lower() != 'ready' and time_waited < 120:
        print(f"Waiting for collection {collection_name} to be ready. Waited {time_waited} seconds...")
        time.sleep(5)
        time_waited += 5
        desc = pc.describe_collection(collection_name)
        collection_ready = desc['status']

    if time_waited >= 120:
        raise Exception(f"Collection {collection_name} is not ready after 120 seconds")

    yield collection_name

    print(f"Deleting collection {collection_name}...")
    pc.delete_collection(collection_name)

@pytest.fixture(autouse=True)
def cleanup(client, index_name):
    yield

    time_waited = 0
    while index_exists(index_name, client) and time_waited < 120:
        print(f"Waiting for index {index_name} to be ready to delete. Waited {time_waited} seconds..")
        time_waited += 5
        time.sleep(5)
        try:
            print(f"Attempting delete of index {index_name}")
            client.delete_index(index_name, -1)
            print(f"Deleted index {index_name}")
            break
        except Exception as e:
            print(f"Unable to delete index {index_name}: {e}")
            pass

    if time_waited >= 120:
        raise Exception(f"Index {index_name} could not be deleted after 120 seconds")
