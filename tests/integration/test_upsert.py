import pytest
from pinecone import Vector, SparseValues

class TestUpsert:
    def test_upsert_sanity(self, client, ready_sl_index, random_vector):
        idx = client.Index(ready_sl_index)

        # Tuples
        idx.upsert(vectors=[('1', random_vector), ('2', random_vector), ('3', random_vector)])

        # Tuples with metadata
        idx.upsert(vectors=[('4', random_vector, {'key': 'value'}), ('5', random_vector, {'key': 'value2'})])

        # Vector objects
        idx.upsert(vectors=[Vector(id='6', values=random_vector)])
        idx.upsert(vectors=[Vector(id='7', values=random_vector, metadata={'key': 'value'})])

        # Dict
        idx.upsert(vectors=[{'id': '8', 'values': random_vector}])

        # Dict with metadata
        idx.upsert(vectors=[{'id': '8', 'values': random_vector, 'metadata': {'key': 'value'}}])

        idx.describe_index_stats()

    def test_upsert_sparse_vectors(self, client, random_vector, create_sl_index_params, index_name):
        create_sl_index_params['metric'] = 'dotproduct'
        create_sl_index_params['timeout'] = 300
        client.create_index(**create_sl_index_params)

        idx = client.Index(index_name)
        idx.upsert(vectors=[Vector(id='1', values=random_vector, sparse_values=SparseValues(values=[0.1, 0.2, 0.3], indices=[1, 2, 3]))])
        idx.upsert(vectors=[{'id': '8', 'values': random_vector, 'metadata': {'key': 'value'}, 'sparse_values': {'values': [0.1, 0.2, 0.3], 'indices': [1, 2, 3] }}])

    def test_upsert_with_invalid_vector(self, client, ready_sl_index, random_vector):
        idx = client.Index(ready_sl_index)

        with pytest.raises(TypeError):
            # non-vector
            idx.upsert(vectors=[('1', 'invalid_vector')])

        with pytest.raises(TypeError):
            # bogus metadata
            idx.upsert(vectors=[('1', random_vector, 'invalid_metadata')])

        with pytest.raises(TypeError):
            # non-string id
            idx.upsert(vectors=[(1, random_vector)])

        with pytest.raises(TypeError):
            idx.upsert(vectors=[{'id': 1, 'values': random_vector}])

    