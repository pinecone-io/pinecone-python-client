import pytest
import os
from pinecone import Vector, SparseValues
from .utils import embedding_values
from ..helpers import fake_api_key
from pinecone import PineconeException

class TestUpsertApiKeyMissing():
    def test_upsert_fails_when_api_key_invalid(self, index_name, index_host):
        with pytest.raises(PineconeException):
            from pinecone import Pinecone
            pc = Pinecone(api_key=fake_api_key())
            idx = pc.Index(name=index_name, host=index_host)
            idx.upsert(vectors=[
                    Vector(id='1', values=embedding_values()),
                    Vector(id='2', values=embedding_values())
                ]
            )
    @pytest.mark.skipif(os.getenv('USE_GRPC') != 'true', reason='Only test grpc client when grpc extras')
    def test_upsert_fails_when_api_key_invalid_grpc(self, index_name, index_host):
        with pytest.raises(PineconeException):
            from pinecone.grpc import PineconeGRPC
            pc = PineconeGRPC(api_key=fake_api_key())
            idx = pc.Index(name=index_name, host=index_host)
            idx.upsert(vectors=[
                    Vector(id='1', values=embedding_values()),
                    Vector(id='2', values=embedding_values())
                ]
            )

class TestUpsertFailsWhenDimensionMismatch():
    def test_upsert_fails_when_dimension_mismatch_objects(self, idx):
        with pytest.raises(PineconeException):
            idx.upsert(vectors=[
                    Vector(id='1', values=embedding_values(2)),
                    Vector(id='2', values=embedding_values(3))
                ])
            
    def test_upsert_fails_when_dimension_mismatch_tuples(self, idx):
        with pytest.raises(PineconeException):
            idx.upsert(vectors=[
                    ('1', embedding_values(2)),
                    ('2', embedding_values(3))
                ])
    
    def test_upsert_fails_when_dimension_mismatch_dicts(self, idx):
        with pytest.raises(PineconeException):
            idx.upsert(vectors=[
                    {'id': '1', 'values': embedding_values(2)},
                    {'id': '2', 'values': embedding_values(3)}
                ])

@pytest.mark.skipif(os.getenv('METRIC') != 'dotproduct', reason='Only metric=dotprodouct indexes support hybrid')
class TestUpsertFailsSparseValuesDimensionMismatch():  
    def test_upsert_fails_when_sparse_values_indices_values_mismatch_objects(self, idx):
        with pytest.raises(PineconeException):
            idx.upsert(vectors=[
                    Vector(id='1', values=[0.1, 0.1], sparse_values=SparseValues(indices=[0], values=[0.5, 0.5]))
                ])
        with pytest.raises(PineconeException):
            idx.upsert(vectors=[
                    Vector(id='1', values=[0.1, 0.1], sparse_values=SparseValues(indices=[0, 1], values=[0.5]))
                ])
            
    def test_upsert_fails_when_sparse_values_in_tuples(self, idx):
        with pytest.raises(ValueError):
            idx.upsert(vectors=[
                    ('1', SparseValues(indices=[0], values=[0.5])),
                    ('2', SparseValues(indices=[0, 1, 2], values=[0.5, 0.5, 0.5]))
                ])
        
    def test_upsert_fails_when_sparse_values_indices_values_mismatch_dicts(self, idx):
        with pytest.raises(PineconeException):
            idx.upsert(vectors=[
                    {'id': '1', 'values': [0.2, 0.2], 'sparse_values': SparseValues(indices=[0], values=[0.5, 0.5])}
                ])
        with pytest.raises(PineconeException):
            idx.upsert(vectors=[
                    {'id': '1', 'values': [0.1, 0.2], 'sparse_values': SparseValues(indices=[0, 1], values=[0.5])}
                ])

class TestUpsertFailsWhenValuesMissing():
    def test_upsert_fails_when_values_missing_objects(self, idx):
        with pytest.raises(TypeError):
            idx.upsert(vectors=[
                    Vector(id='1'),
                    Vector(id='2')
                ])
            
    def test_upsert_fails_when_values_missing_tuples(self, idx):
        with pytest.raises(ValueError):
            idx.upsert(vectors=[
                    ('1',),
                    ('2',)
                ])
            
    def test_upsert_fails_when_values_missing_dicts(self, idx):
        with pytest.raises(ValueError):
            idx.upsert(vectors=[
                    {'id': '1'},
                    {'id': '2'}
                ])
            
class TestUpsertFailsWhenValuesWrongType():
    def test_upsert_fails_when_values_wrong_type_objects(self, idx):
        with pytest.raises(PineconeException):
            idx.upsert(vectors=[
                    Vector(id='1', values='abc'),
                    Vector(id='2', values='def')
                ])
    
    def test_upsert_fails_when_values_wrong_type_tuples(self, idx):
        if os.environ.get('USE_GRPC', 'false') == 'true':
            expected_exception = TypeError
        else:
            expected_exception = PineconeException

        with pytest.raises(expected_exception):
            idx.upsert(vectors=[
                    ('1', 'abc'),
                    ('2', 'def')
                ])
            
    def test_upsert_fails_when_values_wrong_type_dicts(self, idx):
        with pytest.raises(TypeError):
            idx.upsert(vectors=[
                    {'id': '1', 'values': 'abc'},
                    {'id': '2', 'values': 'def'}
                ])

class TestUpsertFailsWhenVectorsMissing():
    def test_upsert_fails_when_vectors_empty(self, idx):
        with pytest.raises(PineconeException):
            idx.upsert(vectors=[])

    def test_upsert_fails_when_vectors_wrong_type(self, idx):
        with pytest.raises(ValueError):
            idx.upsert(vectors='abc')

    def test_upsert_fails_when_vectors_missing(self, idx):
        with pytest.raises(TypeError):
            idx.upsert()

class TestUpsertIdMissing():
    def test_upsert_fails_when_id_is_missing_objects(self, idx):
        with pytest.raises(TypeError):
            idx.upsert(vectors=[
                    Vector(id='1', values=embedding_values()),
                    Vector(values=embedding_values())
                ])
            
    def test_upsert_fails_when_id_is_missing_tuples(self, idx):
        with pytest.raises(ValueError):
            idx.upsert(vectors=[
                    ('1', embedding_values()),
                    (embedding_values())
                ])
            
    def test_upsert_fails_when_id_is_missing_dicts(self, idx):
        with pytest.raises(ValueError):
            idx.upsert(vectors=[
                    {'id': '1', 'values': embedding_values()},
                    {'values': embedding_values()}
                ])


class TestUpsertIdWrongType():
    def test_upsert_fails_when_id_wrong_type_objects(self, idx):
        with pytest.raises(Exception):
            idx.upsert(vectors=[
                    Vector(id='1', values=embedding_values()),
                    Vector(id=2, values=embedding_values())
                ])
            
    def test_upsert_fails_when_id_wrong_type_tuples(self, idx):
        with pytest.raises(Exception):
            idx.upsert(vectors=[
                    ('1', embedding_values()),
                    (2, embedding_values())
                ])
            
    def test_upsert_fails_when_id_wrong_type_dicts(self, idx):
        with pytest.raises(Exception):
            idx.upsert(vectors=[
                    {'id': '1', 'values': embedding_values()},
                    {'id': 2, 'values': embedding_values()}
                ])