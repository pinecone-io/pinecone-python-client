import pytest
import os
import time
import random
import string
from pinecone import Pinecone, ServerlessSpec

def random_string(length):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))

@pytest.fixture
def index_name():
    return 'test-sanity-rest-' + random_string(20)

@pytest.fixture
def client():
    return Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

@pytest.fixture(autouse=True)
def cleanup(index_name, client):
    yield
    if index_name in client.list_indexes().names():
        client.delete_index(name=index_name)

class TestSanityRest:
    def test_sanity(self, index_name, client):
        if index_name not in client.list_indexes().names():
            client.create_index(
                name=index_name, 
                dimension=2, 
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
        else:
            pytest.fail('Index ' + index_name + ' already exists')
        
        idx = client.Index(index_name)
        idx.upsert(vectors=[
            ('1', [1.0, 2.0]), 
            ('2', [3.0, 4.0]),
            ('3', [5.0, 6.0])
        ])

        # Wait for index freshness
        time.sleep(60)

        # Check the vector count reflects the upserted data
        description = idx.describe_index_stats()
        assert description.dimension == 2
        assert description.total_vector_count == 3

        # Query for results
        query_results = idx.query(id='1', top_k=10, include_values=True)
        assert query_results.matches[0].id == '1'
        assert len(query_results.matches) == 3