import pytest
from pinecone import PineconeException, FetchResponse

class TestFetch:
    def setup_method(self):
        self.expected_dimension = 2

    @pytest.mark.parametrize('use_nondefault_namespace', [True, False]) 
    def test_fetch_multiple_by_id(
        self,
        idx, 
        namespace,
        use_nondefault_namespace
    ):
        target_namespace = namespace if use_nondefault_namespace else ''

        results = idx.fetch(ids=['1', '2', '4'], namespace=target_namespace)
        assert isinstance(results, FetchResponse) == True

        assert results.usage != None
        assert results.usage['read_units'] != None
        assert results.usage['read_units'] > 0

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
        assert len(results.vectors['1'].values) == self.expected_dimension


    @pytest.mark.parametrize('use_nondefault_namespace', [True, False]) 
    def test_fetch_single_by_id(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        results = idx.fetch(ids=['1'], namespace=target_namespace)
        assert results.namespace == target_namespace
        assert len(results.vectors) == 1
        assert results.vectors['1'].id == '1'
        assert results.vectors['1'].metadata == None
        assert results.vectors['1'].values != None
        assert len(results.vectors['1'].values) == self.expected_dimension

    @pytest.mark.parametrize('use_nondefault_namespace', [True, False]) 
    def test_fetch_nonexistent_id(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        # Fetch id that is missing
        results = idx.fetch(ids=['100'], namespace=target_namespace)
        assert results.namespace == target_namespace
        assert len(results.vectors) == 0

    def test_fetch_nonexistent_namespace(self, idx):
        target_namespace = 'nonexistent-namespace'

        # Fetch from namespace with no vectors
        results = idx.fetch(ids=['1'], namespace=target_namespace)
        assert results.namespace == target_namespace
        assert len(results.vectors) == 0

    @pytest.mark.parametrize('use_nondefault_namespace', [True, False]) 
    def test_fetch_with_empty_list_of_ids(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        # Fetch with empty list of ids
        with pytest.raises(PineconeException) as e:
            idx.fetch(ids=[], namespace=target_namespace)
        assert 'ids' in str(e.value).lower()

    def test_fetch_unspecified_namespace(self, idx):
        # Fetch without specifying namespace gives default namespace results
        results = idx.fetch(ids=['1', '4'])
        assert results.namespace == ''
        assert results.vectors['1'].id == '1'
        assert results.vectors['1'].values != None
        assert results.vectors['4'].metadata != None

