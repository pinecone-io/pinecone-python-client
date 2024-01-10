import pytest
import os
import json
from ..helpers import get_environment_var, random_string

# Test matrix needs to consider the following dimensions:
# - pod vs serverless
# - grpc vs rest
# - metric -> vector vs sparse vector
# - namespace: default vs custom
# - environment: free vs paid
# - with metadata vs without metadata

@pytest.fixture
def api_key():
    return get_environment_var('PINECONE_API_KEY')

@pytest.fixture
def client(api_key):
    use_grpc = os.environ.get('USE_GRPC', 'false') == 'true'
    if use_grpc:
        from pinecone.grpc import PineconeGRPC
        return PineconeGRPC(api_key=api_key)
    else:
        from pinecone import Pinecone
        return Pinecone(api_key=api_key)
    
@pytest.fixture
def metric():
    return get_environment_var('METRIC', 'cosine')

@pytest.fixture
def spec():
    return json.loads(get_environment_var('SPEC'))

@pytest.fixture
def index_name():
    return 'dataplane-' + random_string(20)
    
@pytest.fixture
def namespace():
    return random_string(10)

@pytest.fixture
def index_host(client, index_name, metric, spec):
    client.create_index(name=index_name, dimension=2, metric=metric, spec=spec)
    description = client.describe_index(name=index_name)
    return description.host

def sleep_t():
    return int(os.environ.get('FRESHNESS_SLEEP_SECONDS', 60))