import pytest
from pinecone import PineconeException

@pytest.mark.parametrize('use_nondefault_namespace', [True, False])
class TestQueryErrorCases:
    def test_query_with_invalid_vector(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        with pytest.raises(PineconeException) as e:
            idx.query(vector=[1, 2, 3], namespace=target_namespace, top_k=10)
        
        assert 'vector' in str(e.value).lower()

    def test_query_with_invalid_id(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        with pytest.raises(TypeError) as e:
            idx.query(id=1, namespace=target_namespace, top_k=10)

    def test_query_with_invalid_top_k(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        with pytest.raises((PineconeException, ValueError)) as e:
            idx.query(id='1', namespace=target_namespace, top_k=-1)

    def test_query_with_missing_top_k(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        with pytest.raises((TypeError, PineconeException)) as e:
            idx.query(id='1', namespace=target_namespace)

        assert 'top_k' in str(e.value).lower()
    
