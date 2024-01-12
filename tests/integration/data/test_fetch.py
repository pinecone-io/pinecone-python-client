import pytest
from pinecone import Vector, PineconeException, FetchResponse
from ..helpers import poll_fetch_for_ids_in_namespace
from .utils import embedding_values
2
@pytest.mark.parametrize('use_nondefault_namespace', [True, False]) 
def test_upsert_to_namespace(
    idx, 
    namespace,
    use_nondefault_namespace
):
    target_namespace = namespace if use_nondefault_namespace else ''
    expected_dimension = 2

    def setup_data(idx, target_namespace):
        # Upsert without metadata
        idx.upsert(vectors=[
                ('1', embedding_values(2)), 
                ('2', embedding_values(2)),
                ('3', embedding_values(2))
            ], 
            namespace=target_namespace
        )

        # Upsert with metadata
        idx.upsert(vectors=[
                Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
                Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
                Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
            ], 
            namespace=target_namespace
        )

        # Upsert with dict
        idx.upsert(vectors=[
                {'id': '7', 'values': embedding_values(2)},
                {'id': '8', 'values': embedding_values(2)},
                {'id': '9', 'values': embedding_values(2)}
            ], 
            namespace=target_namespace
        )

    def fetch_multiple_by_id(idx, target_namespace):
        # Fetch multiple by id
        results = idx.fetch(ids=['1', '2', '4'], namespace=target_namespace)
        assert isinstance(results, FetchResponse) == True
        assert results.namespace == target_namespace
        assert len(results.vectors) == 3
        assert results.vectors['1'].id == '1'
        assert results.vectors['2'].id == '2'
        # Metadata included, if set
        assert results.vectors['1'].metadata == None
        assert results.vectors['2'].metadata == None
        assert results.vectors['4'].metadata != None
        assert results.vectors['4'].metadata['genre'] == 'action'
        assert results.vectors['4'].metadata['runtime'] == 120
        # Values included
        assert results.vectors['1'].values != None
        assert len(results.vectors['1'].values) == expected_dimension

    def fetch_single_by_id(idx, target_namespace):
        # Fetch single by id
        results = idx.fetch(ids=['1'], namespace=target_namespace)
        assert results.namespace == target_namespace
        assert len(results.vectors) == 1
        assert results.vectors['1'].id == '1'
        assert results.vectors['1'].metadata == None
        assert results.vectors['1'].values != None
        assert len(results.vectors['1'].values) == expected_dimension

    def fetch_nonexistent_id(idx, target_namespace):
        # Fetch id that is missing
        results = idx.fetch(ids=['100'], namespace=target_namespace)
        assert results.namespace == target_namespace
        assert len(results.vectors) == 0

    def fetch_nonexistent_namespace(idx):
        # Fetch from namespace with no vectors
        results = idx.fetch(ids=['1'], namespace='nonexistent-namespace')
        assert results.namespace == 'nonexistent-namespace'
        assert len(results.vectors) == 0

    def fetch_with_empty_list_of_ids(idx, target_namespace):
        # Fetch with empty list of ids
        with pytest.raises(PineconeException) as e:
            idx.fetch(ids=[], namespace=target_namespace)
        assert 'ids' in str(e.value).lower()

    def fetch_unspecified_namespace(idx):
        # Fetch without specifying namespace gives default namespace results
        results = idx.fetch(ids=['1', '4'])
        assert results.namespace == ''
        assert results.vectors['1'].id == '1'
        assert results.vectors['1'].values != None
        assert results.vectors['4'].metadata != None

    # Using an unconventional test arrangement here because
    # I want to minimize the amount of waiting for data 
    # freshness in the test suite. So we upsert all the data
    # up front and then run a bunch of fetch tests.
        
    setup_data(idx, target_namespace)
    poll_fetch_for_ids_in_namespace(idx, ids=['1', '2', '3', '4', '5', '6', '7', '8', '9'], namespace=target_namespace)
    
    fetch_multiple_by_id(idx, target_namespace)
    fetch_single_by_id(idx, target_namespace)
    fetch_nonexistent_id(idx, target_namespace)
    fetch_nonexistent_namespace(idx)
    fetch_with_empty_list_of_ids(idx, target_namespace)
    if target_namespace == '':
        fetch_unspecified_namespace(idx)