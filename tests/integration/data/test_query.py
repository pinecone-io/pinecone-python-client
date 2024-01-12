import pytest
from pinecone import QueryResponse
from .utils import embedding_values

def find_by_id(matches, id):
    with_id = [match for match in matches if match.id == id]
    return with_id[0] if len(with_id) > 0 else None

@pytest.mark.parametrize('use_nondefault_namespace', [True, False])
class TestQuery: 
    def setup_method(self):
        self.expected_dimension = 2

    def test_query_by_id(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        results = idx.query(id='1', namespace=target_namespace, top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == target_namespace
        
        assert results.usage != None
        assert results.usage['read_units'] != None
        assert results.usage['read_units'] > 0

        # By default, does not include values or metadata
        record_with_metadata = find_by_id(results.matches, '4')
        assert record_with_metadata.metadata == None
        assert record_with_metadata.values == []

    def test_query_by_vector(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        results = idx.query(vector=embedding_values(2), namespace=target_namespace, top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == target_namespace

    def test_query_by_vector_include_values(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        results = idx.query(vector=embedding_values(2), namespace=target_namespace, include_values=True, top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == target_namespace
        assert len(results.matches) > 0
        assert results.matches[0].values != None
        assert len(results.matches[0].values) == self.expected_dimension

    def test_query_by_vector_include_metadata(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        results = idx.query(vector=embedding_values(2), namespace=target_namespace, include_metadata=True, top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == target_namespace

        matches_with_metadata = [match for match in results.matches if match.metadata != None]
        assert len(matches_with_metadata) == 3
        assert find_by_id(results.matches, '4').metadata['genre'] == 'action'

    def test_query_by_vector_include_values_and_metadata(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        results = idx.query(vector=embedding_values(2), namespace=target_namespace, include_values=True, include_metadata=True, top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == target_namespace

        matches_with_metadata = [match for match in results.matches if match.metadata != None]
        assert len(matches_with_metadata) == 3
        assert find_by_id(results.matches, '4').metadata['genre'] == 'action'
        assert len(results.matches[0].values) == self.expected_dimension

class TestQueryEdgeCases():
    def test_query_in_empty_namespace(self, idx):
        results = idx.query(id='1', namespace='empty', top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == 'empty'
        assert len(results.matches) == 0

@pytest.mark.parametrize('use_nondefault_namespace', [True, False])
class TestQueryWithFilter():
    def test_query_by_id_with_filter(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        results = idx.query(id='1', namespace=target_namespace, filter={'genre': 'action'}, top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == target_namespace
        assert len(results.matches) == 1
        assert results.matches[0].id == '4'

    def test_query_by_id_with_filter_gt(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        results = idx.query(id='1', namespace=target_namespace, filter={'runtime': {'$gt': 100}}, top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == target_namespace
        assert len(results.matches) == 2
        assert find_by_id(results.matches, '4') != None
        assert find_by_id(results.matches, '6') != None

    def test_query_by_id_with_filter_gte(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        results = idx.query(id='1', namespace=target_namespace, filter={'runtime': {'$gte': 90}}, top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == target_namespace
        assert len(results.matches) == 3
        assert find_by_id(results.matches, '4') != None
        assert find_by_id(results.matches, '5') != None
        assert find_by_id(results.matches, '6') != None

    def test_query_by_id_with_filter_lt(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        results = idx.query(id='1', namespace=target_namespace, filter={'runtime': {'$lt': 100}}, top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == target_namespace
        assert len(results.matches) == 1
        assert find_by_id(results.matches, '5') != None

    def test_query_by_id_with_filter_lte(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        results = idx.query(id='1', namespace=target_namespace, filter={'runtime': {'$lte': 120}}, top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == target_namespace
        assert len(results.matches) == 2
        assert find_by_id(results.matches, '4') != None
        assert find_by_id(results.matches, '5') != None

    def test_query_by_id_with_filter_in(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        results = idx.query(id='1', namespace=target_namespace, filter={'genre': {'$in': ['romance']}}, top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == target_namespace
        assert len(results.matches) == 1
        assert find_by_id(results.matches, '6') != None

    @pytest.mark.skip(reason='Seems like a bug in the server')
    def test_query_by_id_with_filter_nin(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        results = idx.query(id='1', namespace=target_namespace, filter={'genre': {'$nin': ['romance']}}, top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == target_namespace
        assert len(results.matches) == 2
        assert find_by_id(results.matches, '4') != None
        assert find_by_id(results.matches, '5') != None

    def test_query_by_id_with_filter_eq(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        results = idx.query(id='1', namespace=target_namespace, filter={'genre': {'$eq': 'action'}}, top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == target_namespace
        assert len(results.matches) == 1
        assert find_by_id(results.matches, '4') != None

    @pytest.mark.skip(reason='Seems like a bug in the server')
    def test_query_by_id_with_filter_ne(self, idx, namespace, use_nondefault_namespace):
        target_namespace = namespace if use_nondefault_namespace else ''

        # Vector(id='4', values=embedding_values(2), metadata={'genre': 'action', 'runtime': 120 }),
        # Vector(id='5', values=embedding_values(2), metadata={'genre': 'comedy', 'runtime': 90 }),
        # Vector(id='6', values=embedding_values(2), metadata={'genre': 'romance', 'runtime': 240 })
        results = idx.query(id='1', namespace=target_namespace, filter={'genre': {'$ne': 'action'}}, top_k=10)
        assert isinstance(results, QueryResponse) == True
        assert results.namespace == target_namespace
        assert len(results.matches) == 2
        assert find_by_id(results.matches, '5') != None
        assert find_by_id(results.matches, '6') != None
