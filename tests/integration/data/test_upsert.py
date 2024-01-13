import pytest
import os
from pinecone import Vector, SparseValues
from ..helpers import poll_stats_for_namespace
from .utils import embedding_values

@pytest.mark.parametrize('use_nondefault_namespace', [True, False]) 
def test_upsert_to_namespace(
    idx, 
    namespace,
    use_nondefault_namespace
):
    target_namespace = namespace if use_nondefault_namespace else ''
    
    # Upsert with tuples
    idx.upsert(vectors=[
            ('1', embedding_values()), 
            ('2', embedding_values()),
            ('3', embedding_values())
        ], 
        namespace=target_namespace
    )

    # Upsert with objects
    idx.upsert(vectors=[
            Vector(id='4', values=embedding_values()),
            Vector(id='5', values=embedding_values()),
            Vector(id='6', values=embedding_values())
        ], 
        namespace=target_namespace
    )

    # Upsert with dict
    idx.upsert(vectors=[
            {'id': '7', 'values': embedding_values()},
            {'id': '8', 'values': embedding_values()},
            {'id': '9', 'values': embedding_values()}
        ], 
        namespace=target_namespace
    )

    poll_stats_for_namespace(idx, target_namespace, 9)

    # Check the vector count reflects some data has been upserted
    stats = idx.describe_index_stats()
    assert stats.total_vector_count >= 9
    assert stats.namespaces[target_namespace].vector_count == 9

@pytest.mark.parametrize('use_nondefault_namespace', [True, False]) 
@pytest.mark.skipif(os.getenv('METRIC') != 'dotproduct', reason='Only metric=dotprodouct indexes support hybrid')
def test_upsert_to_namespace_with_sparse_embedding_values(
    idx,
    namespace,
    use_nondefault_namespace
):
    target_namespace = namespace if use_nondefault_namespace else ''

    # Upsert with sparse values object
    idx.upsert(vectors=[
            Vector(
                id='1',
                values=embedding_values(),
                sparse_values=SparseValues(
                    indices=[0,1], 
                    values=embedding_values()
                )
            ),
        ],
        namespace=target_namespace
    )

    # Upsert with sparse values dict
    idx.upsert(vectors=[
            {'id': '2', 'values': embedding_values(),'sparse_values': {'indices': [0,1], 'values': embedding_values()}},
            {'id': '3', 'values': embedding_values(), 'sparse_values': {'indices': [0,1], 'values': embedding_values()}}
        ],
        namespace=target_namespace
    )

    poll_stats_for_namespace(idx, target_namespace, 9)

    # Check the vector count reflects some data has been upserted
    stats = idx.describe_index_stats()
    assert stats.total_vector_count >= 9
    assert stats.namespaces[target_namespace].vector_count == 9

