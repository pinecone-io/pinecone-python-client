import pytest
import os
from pinecone import Vector, SparseValues
from ...helpers import poll_stats_for_namespace
from .utils import embedding_values
from pinecone import PineconeApiTypeError, PineconeApiException

class TestUpsertFailsWhenDimensionMismatch():
    def test_upsert_fails_when_dimension_mismatch_objects(self, idx):
        with pytest.raises(PineconeApiException):
            idx.upsert(vectors=[
                    Vector(id='1', values=embedding_values(2)),
                    Vector(id='2', values=embedding_values(3))
                ])
            
    def test_upsert_fails_when_dimension_mismatch_tuples(self, idx):
        with pytest.raises(PineconeApiException):
            idx.upsert(vectors=[
                    ('1', embedding_values(2)),
                    ('2', embedding_values(3))
                ])
    
    def test_upsert_fails_when_dimension_mismatch_dicts(self, idx):
        with pytest.raises(PineconeApiException):
            idx.upsert(vectors=[
                    {'id': '1', 'values': embedding_values(2)},
                    {'id': '2', 'values': embedding_values(3)}
                ])

@pytest.mark.skipif(os.getenv('METRIC') != 'dotproduct', reason='Only metric=dotprodouct indexes support hybrid')
class TestUpsertFailsSparseValuesDimensionMismatch():
    def test_upsert_fails_when_sparse_values_indices_out_of_range_objects(self, idx):
        with pytest.raises(PineconeApiException):
            idx.upsert(vectors=[
                    Vector(id='1', values=[0.1, 0.1], sparse_values=SparseValues(indices=[0], values=[0.5])),
                    Vector(id='2', values=[0.1, 0.1], sparse_values=SparseValues(indices=[0, 1, 2], values=[0.5, 0.5, 0.5]))
                ])
            
    def test_upsert_fails_when_sparse_values_indices_values_mismatch_objects(self, idx):
        with pytest.raises(PineconeApiException):
            idx.upsert(vectors=[
                    Vector(id='1', values=[0.1, 0.1], sparse_values=SparseValues(indices=[0], values=[0.5, 0.5]))
                ])
        with pytest.raises(PineconeApiException):
            idx.upsert(vectors=[
                    Vector(id='1', values=[0.1, 0.1], sparse_values=SparseValues(indices=[0, 1], values=[0.5]))
                ])
            
    def test_upsert_fails_when_sparse_values_indices_out_of_range_tuples(self, idx):
        with pytest.raises(ValueError):
            idx.upsert(vectors=[
                    ('1', SparseValues(indices=[0], values=[0.5])),
                    ('2', SparseValues(indices=[0, 1, 2], values=[0.5, 0.5, 0.5]))
                ])
            
    def test_upsert_fails_when_sparse_values_indices_values_mismatch_tuples(self, idx):
        with pytest.raises(ValueError):
            idx.upsert(vectors=[
                    ('1', SparseValues(indices=[0], values=[0.5, 0.5]))
                ])
        with pytest.raises(ValueError):
            idx.upsert(vectors=[
                    ('1', SparseValues(indices=[0, 1], values=[0.5]))
                ])
            
    def test_upsert_fails_when_sparse_values_indices_out_of_range_dicts(self, idx):
        with pytest.raises(PineconeApiException):
            idx.upsert(vectors=[
                    {'id': '1', 'values': [], 'sparse_values': SparseValues(indices=[0], values=[0.5])},
                    {'id': '2', 'values': [], 'sparse_values': SparseValues(indices=[0, 1, 2], values=[0.5, 0.5, 0.5])}
                ])
            
    def test_upsert_fails_when_sparse_values_indices_values_mismatch_dicts(self, idx):
        with pytest.raises(PineconeApiException):
            idx.upsert(vectors=[
                    {'id': '1', 'values': [], 'sparse_values': SparseValues(indices=[0], values=[0.5, 0.5])}
                ])
        with pytest.raises(PineconeApiException):
            idx.upsert(vectors=[
                    {'id': '1', 'values': [], 'sparse_values': SparseValues(indices=[0, 1], values=[0.5])}
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
        with pytest.raises(PineconeApiTypeError):
            idx.upsert(vectors=[
                    Vector(id='1', values='abc'),
                    Vector(id='2', values='def')
                ])
            
    def test_upsert_fails_when_values_wrong_type_tuples(self, idx):
        with pytest.raises(PineconeApiTypeError):
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
        with pytest.raises(PineconeApiException):
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
        with pytest.raises(PineconeApiTypeError):
            idx.upsert(vectors=[
                    Vector(id='1', values=embedding_values()),
                    Vector(id=2, values=embedding_values())
                ])
            
    def test_upsert_fails_when_id_wrong_type_tuples(self, idx):
        with pytest.raises(PineconeApiTypeError):
            idx.upsert(vectors=[
                    ('1', embedding_values()),
                    (2, embedding_values())
                ])
            
    def test_upsert_fails_when_id_wrong_type_dicts(self, idx):
        with pytest.raises(PineconeApiTypeError):
            idx.upsert(vectors=[
                    {'id': '1', 'values': embedding_values()},
                    {'id': 2, 'values': embedding_values()}
                ])