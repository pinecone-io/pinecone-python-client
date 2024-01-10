import pytest
import time
from pinecone import Vector

def test_upsert_to_default_namespace(client, index_name, sleep_t):
    expected_dimension = 2
    desc = client.describe_index(index_name)
    assert desc.dimension == expected_dimension
    assert desc.metric == 'cosine'

    idx = client.Index(index_name)
    
    # Upsert with tuples
    idx.upsert(vectors=[
        ('1', [1.0, 2.0]), 
        ('2', [3.0, 4.0]),
        ('3', [5.0, 6.0])
    ])

    # Upsert with objects
    idx.upsert(vectors=[
        Vector('4', [7.0, 8.0]),
        Vector('5', [9.0, 10.0]),
        Vector('6', [11.0, 12.0])
    ])

    # Upsert with dict
    idx.upsert(vectors=[
        {'id': '7', 'values': [13.0, 14.0]},
        {'id': '8', 'values': [15.0, 16.0]},
        {'id': '9', 'values': [17.0, 18.0]}
    ])

    time.sleep(sleep_t)

    # Check the vector count reflects some data has been upserted
    stats = idx.describe_index_stats()
    assert stats.vector_count == 9
    

def test_upsert_to_custom_namespace(client, index_name, namespace):
    expected_dimension = 2
    assert client.describe_index(index_name).dimension == expected_dimension

    idx = client.Index(index_name)
    
    # Upsert with tuples
    idx.upsert(vectors=[
        ('1', [1.0, 2.0]), 
        ('2', [3.0, 4.0]),
        ('3', [5.0, 6.0])
        ], 
        namespace=namespace
    )
