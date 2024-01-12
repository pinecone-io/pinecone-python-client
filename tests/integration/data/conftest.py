import pytest
import os
import time
import json
from ..helpers import get_environment_var, random_string
from .seed import setup_data

# Test matrix needs to consider the following dimensions:
# - pod vs serverless
# - grpc vs rest
# - metric -> vector vs sparse vector
# - namespace: default vs custom
# - environment: free vs paid
# - with metadata vs without metadata

def api_key():
    return get_environment_var('PINECONE_API_KEY')

def use_grpc():
    return os.environ.get('USE_GRPC', 'false') == 'true'

def build_client():
    if use_grpc():
        from pinecone.grpc import PineconeGRPC
        return PineconeGRPC(api_key=api_key())
    else:
        from pinecone import Pinecone
        return Pinecone(api_key=api_key())

@pytest.fixture(scope='session')
def client():
    return build_client()
    
@pytest.fixture(scope='session')
def metric():
    return get_environment_var('METRIC', 'cosine')

@pytest.fixture(scope='session')
def spec():
    return json.loads(get_environment_var('SPEC'))

@pytest.fixture(scope='session')
def index_name():
    # return 'dataplane-lol'
    return 'dataplane-' + random_string(20)
    
@pytest.fixture(scope='session')
def namespace():
    # return 'banana'
    return random_string(10)

@pytest.fixture(scope='session')
def idx(client, index_name, index_host):
    return client.Index(name=index_name, host=index_host)

@pytest.fixture(scope='session')
def index_host(index_name, metric, spec):
    pc = build_client()
    print('Creating index with name: ' + index_name)
    pc.create_index(
        name=index_name, 
        dimension=2, 
        metric=metric, 
        spec=spec
    )
    description = pc.describe_index(name=index_name)
    yield description.host
    print('Deleting index with name: ' + index_name)
    pc.delete_index(index_name, -1)

@pytest.fixture(scope='session', autouse=True)
def seed_data(idx, namespace, index_host):
    print('Seeding data in host ' + index_host)

    print('Seeding data in namespace "' + namespace + '"')
    setup_data(idx, namespace, False)

    print('Seeding data in namespace ""')
    setup_data(idx, '', True)

    print('Waiting a bit more to ensure freshness')
    time.sleep(60)
    yield