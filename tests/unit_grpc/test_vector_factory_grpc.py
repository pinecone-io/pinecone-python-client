import numpy as np
import pandas as pd
import pytest

from collections.abc import Iterable, Mapping

from pinecone.grpc.vector_factory_grpc import VectorFactoryGRPC
from pinecone.grpc import Vector, SparseValues
from pinecone.grpc.utils import dict_to_proto_struct
from pinecone import Vector as NonGRPCVector, SparseValues as NonGRPCSparseValues


class TestVectorFactoryGRPC:
    def test_build_when_returns_vector_unmodified(self):
        vec = Vector(id="1", values=[0.1, 0.2, 0.3])
        assert VectorFactoryGRPC.build(vec) == vec
        assert VectorFactoryGRPC.build(vec).__class__ == Vector

    def test_build_when_nongrpc_vector_it_converts(self):
        vec = NonGRPCVector(id="1", values=[0.1, 0.2, 0.3])
        assert VectorFactoryGRPC.build(vec) == Vector(id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({}))

    def test_build_when_nongrpc_vector_with_metadata_it_converts(self):
        vec = NonGRPCVector(id="1", values=[0.1, 0.2, 0.3], metadata={"genre": "comedy"})
        assert VectorFactoryGRPC.build(vec) == Vector(
            id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({"genre": "comedy"})
        )

    def test_build_when_nongrpc_vector_with_sparse_values_it_converts(self):
        vec = NonGRPCVector(
            id="1", values=[0.1, 0.2, 0.3], sparse_values=NonGRPCSparseValues(indices=[0, 2], values=[0.1, 0.3])
        )
        assert VectorFactoryGRPC.build(vec) == Vector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({}),
            sparse_values=SparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )

    @pytest.mark.parametrize("values_array", [
        [0.1, 0.2, 0.3],
        np.array([0.1, 0.2, 0.3]), 
        pd.array([0.1, 0.2, 0.3])
    ])
    def test_build_when_tuple_with_two_values(self, values_array):
        tup = ("1", values_array)
        actual = VectorFactoryGRPC.build(tup)
        expected = Vector(id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({}))
        assert actual == expected

    @pytest.mark.parametrize("values_array", [
        [0.1, 0.2, 0.3],
        np.array([0.1, 0.2, 0.3]), 
        pd.array([0.1, 0.2, 0.3])
    ])
    def test_build_when_tuple_with_three_values(self, values_array):
        tup = ("1", values_array, {"genre": "comedy"})
        actual = VectorFactoryGRPC.build(tup)
        expected = Vector(id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({"genre": "comedy"}))
        assert actual == expected

    @pytest.mark.parametrize("sv_klass", [SparseValues, NonGRPCSparseValues])
    def test_build_vector_with_tuple_with_sparse_values(self, sv_klass):
        tup = ("1", sv_klass(indices=[0, 2], values=[0.1, 0.3]))
        with pytest.raises(
            ValueError,
            match="Sparse values are not supported in tuples. Please use either dicts or Vector objects as inputs.",
        ):
            VectorFactoryGRPC.build(tup)

    def test_build_when_tuple_errors_when_additional_fields(self):
        with pytest.raises(ValueError, match="Found a tuple of length 4 which is not supported"):
            tup = ("1", [0.1, 0.2, 0.3], {"a": "b"}, "extra")
            VectorFactoryGRPC.build(tup)

    def test_build_when_tuple_too_short(self):
        with pytest.raises(ValueError, match="Found a tuple of length 1 which is not supported"):
            tup = ("1",)
            VectorFactoryGRPC.build(tup)

    @pytest.mark.parametrize("metadata", [
        {"genre": "comedy"},
        dict_to_proto_struct({"genre": "comedy"})]
    )
    def test_build_when_dict(self, metadata):
        d = {"id": "1", "values": [0.1, 0.2, 0.3], "metadata": metadata}
        actual = VectorFactoryGRPC.build(d)
        expected = Vector(id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({"genre": "comedy"}))
        assert actual == expected

    def test_build_with_dict_no_metadata(self):
        d = {"id": "1", "values": [0.1, 0.2, 0.3]}
        actual = VectorFactoryGRPC.build(d)
        expected = Vector(id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({}))
        assert actual == expected

    def test_build_when_dict_with_sparse_values_dict(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": {"indices": [0, 2], "values": [0.1, 0.3]},
        }
        actual = VectorFactoryGRPC.build(d)
        expected = Vector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({"genre": "comedy"}),
            sparse_values=SparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    @pytest.mark.parametrize("sv_klass", [SparseValues, NonGRPCSparseValues])
    def test_build_with_dict_with_sparse_values_object(self, sv_klass):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": sv_klass(indices=[0, 2], values=[0.1, 0.3]),
        }
        actual = VectorFactoryGRPC.build(d)
        expected = Vector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({"genre": "comedy"}),
            sparse_values=SparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    @pytest.mark.parametrize("input_values", [
        pd.array([0.1, 0.2, 0.3]),
        np.array([0.1, 0.2, 0.3])
    ])
    def test_build_when_dict_with_special_values(self, input_values):
        d = {"id": "1", "values": input_values, "metadata": {"genre": "comedy"}}
        actual = VectorFactoryGRPC.build(d)
        expected = Vector(id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({"genre": "comedy"}))
        assert actual == expected

    def test_build_when_dict_missing_required_fields(self):
        with pytest.raises(ValueError, match="Vector dictionary is missing required fields"):
            d = {"values": [0.1, 0.2, 0.3]}
            VectorFactoryGRPC.build(d)

    def test_build_when_dict_excess_keys(self):
        with pytest.raises(ValueError, match="Found excess keys in the vector dictionary"):
            d = {"id": "1", "values": [0.1, 0.2, 0.3], "metadata": {"genre": "comedy"}, "extra": "field"}
            VectorFactoryGRPC.build(d)

    @pytest.mark.parametrize("sv_indices,sv_values", [
        ([0, 2], [0.1, 0.3]),
        (pd.array([0, 2]), [0.1, 0.3]),
        ([0, 2], pd.array([0.1, 0.3])),
        (pd.array([0, 2]), pd.array([0.1, 0.3])),
        (np.array([0, 2]), [0.1, 0.3]),
        ([0, 2], np.array([0.1, 0.3]))
    ])
    def test_build_when_dict_sparse_values(self, sv_indices, sv_values):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": {"indices": sv_indices, "values": sv_values},
        }
        actual = VectorFactoryGRPC.build(d)
        expected = Vector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({"genre": "comedy"}),
            sparse_values=SparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    def test_build_when_dict_sparse_values_when_SparseValues(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": SparseValues(indices=[0, 2], values=[0.1, 0.3]),
        }
        actual = VectorFactoryGRPC.build(d)
        expected = Vector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({"genre": "comedy"}),
            sparse_values=SparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    @pytest.mark.parametrize("bogus_sparse_values", [
        1,
        "not an array",
        [1, 2],
        {}
    ])
    def test_build_when_dict_sparse_values_errors_when_invalid_sparse_values_values(self, bogus_sparse_values):
        with pytest.raises(ValueError, match="Found unexpected data in column `sparse_values`"):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "sparse_values": {"indices": [1, 2], "values": bogus_sparse_values},
            }
            VectorFactoryGRPC.build(d)

    @pytest.mark.parametrize("bogus_sparse_indices", [
        1,
        "not an array",
        [0.1, 0.2],
        {}
    ])
    def test_build_when_dict_sparse_values_errors_when_values_not_list(self, bogus_sparse_indices):
        with pytest.raises(ValueError, match="Found unexpected data in column `sparse_values`"):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "sparse_values": {"indices": bogus_sparse_indices, "values": [0.1, 0.1]},
            }
            VectorFactoryGRPC.build(d)

    def test_build_when_errors_when_other_type(self):
        with pytest.raises(ValueError, match="Invalid vector value passed: cannot interpret type"):
            VectorFactoryGRPC.build(1)

    @pytest.mark.parametrize("bogus_sparse_values", [
        1, 
        "not a dict", 
        [1, 2, 3],
        [],
    ])
    def test_build_when_invalid_sparse_values_type_in_dict(self, bogus_sparse_values):
        with pytest.raises(ValueError, match="Column `sparse_values` is expected to be a dictionary"):
            d = {
                'id': '1', 
                'values': [0.1, 0.2, 0.3], 
                'metadata': {'genre': 'comedy'}, 
                'sparse_values': bogus_sparse_values # not a valid dict
            }
            VectorFactoryGRPC.build(d)

    @pytest.mark.parametrize("bogus_sparse_values", [
        {},
        {'indices': [0, 2]},
        {'values': [0.1, 0.3]},
    ])
    def test_build_when_missing_keys_in_sparse_values_dict(self, bogus_sparse_values):
        with pytest.raises(ValueError, match="Missing required keys in data in column `sparse_values`"):
            d = {
                'id': '1', 
                'values': [0.1, 0.2, 0.3], 
                'metadata': {'genre': 'comedy'}, 
                'sparse_values': bogus_sparse_values
            }
            VectorFactoryGRPC.build(d)

    def test_build_when_sparse_values_is_None(self):
        d = {
            'id': '1',
            'values': [0.1, 0.2, 0.3],
            'metadata': {'genre': 'comedy'},
            'sparse_values': None
        }
        actual = VectorFactoryGRPC.build(d)
        expected = Vector(
            id='1', 
            values=[0.1, 0.2, 0.3], 
            metadata=dict_to_proto_struct({'genre': 'comedy'})
        )
        assert actual == expected