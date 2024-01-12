import numpy as np
import pandas as pd
import pytest

from pinecone.grpc.vector_factory_grpc import VectorFactoryGRPC
from pinecone.grpc import Vector, SparseValues
from pinecone.grpc.utils import dict_to_proto_struct
from pinecone import Vector as NonGRPCVector, SparseValues as NonGRPCSparseValues


class TestVectorFactory:
    def test_build_when_returns_vector_unmodified(self):
        vec = Vector(id="1", values=[0.1, 0.2, 0.3])
        assert VectorFactoryGRPC.build(vec) == vec
        assert VectorFactoryGRPC.build(vec).__class__ == Vector

    def test_build_when_nongrpc_vector(self):
        vec = NonGRPCVector(id="1", values=[0.1, 0.2, 0.3])
        assert VectorFactoryGRPC.build(vec) == Vector(id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({}))

    def test_build_when_nongrpc_vector_with_metadata(self):
        vec = NonGRPCVector(id="1", values=[0.1, 0.2, 0.3], metadata={"genre": "comedy"})
        assert VectorFactoryGRPC.build(vec) == Vector(
            id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({"genre": "comedy"})
        )

    def test_build_when_nongrpc_vector_with_sparse_values(self):
        vec = NonGRPCVector(
            id="1", values=[0.1, 0.2, 0.3], sparse_values=NonGRPCSparseValues(indices=[0, 2], values=[0.1, 0.3])
        )
        assert VectorFactoryGRPC.build(vec) == Vector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({}),
            sparse_values=SparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )

    def test_build_when_tuple_with_two_values(self):
        tup = ("1", [0.1, 0.2, 0.3])
        actual = VectorFactoryGRPC.build(tup)
        expected = Vector(id="1", values=[0.1, 0.2, 0.3], metadata={})
        assert actual == expected

    def test_build_when_tuple_with_three_values(self):
        tup = ("1", [0.1, 0.2, 0.3], {"genre": "comedy"})
        actual = VectorFactoryGRPC.build(tup)
        expected = Vector(id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({"genre": "comedy"}))
        assert actual == expected

    def test_build_when_tuple_with_numpy_array(self):
        tup = ("1", np.array([0.1, 0.2, 0.3]), {"genre": "comedy"})
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

    def test_build_when_tuple_with_pandas_array(self):
        tup = ("1", pd.array([0.1, 0.2, 0.3]))
        actual = VectorFactoryGRPC.build(tup)
        expected = Vector(id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({}))
        assert actual == expected

    def test_build_when_tuple_errors_when_additional_fields(self):
        with pytest.raises(ValueError, match="Found a tuple of length 4 which is not supported"):
            tup = ("1", [0.1, 0.2, 0.3], {"a": "b"}, "extra")
            VectorFactoryGRPC.build(tup)

    def test_build_when_tuple_too_short(self):
        with pytest.raises(ValueError, match="Found a tuple of length 1 which is not supported"):
            tup = ("1",)
            VectorFactoryGRPC.build(tup)

    @pytest.mark.parametrize("metadata", [{"genre": "comedy"}, dict_to_proto_struct({"genre": "comedy"})])
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

    @pytest.mark.parametrize("metadata", [{"genre": "comedy"}, dict_to_proto_struct({"genre": "comedy"})])
    def test_build_when_dict_with_numpy_values(self, metadata):
        d = {"id": "1", "values": np.array([0.1, 0.2, 0.3]), "metadata": metadata}
        actual = VectorFactoryGRPC.build(d)
        expected = Vector(id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({"genre": "comedy"}))
        assert actual == expected

    @pytest.mark.parametrize("metadata", [{"genre": "comedy"}, dict_to_proto_struct({"genre": "comedy"})])
    def test_build_when_dict_with_pandas_values(self, metadata):
        d = {"id": "1", "values": pd.array([0.1, 0.2, 0.3]), "metadata": metadata}
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

    def test_build_when_dict_sparse_values(self):
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

    def test_build_when_dict_sparse_values_errors_when_not_dict(self):
        with pytest.raises(ValueError, match="Column `sparse_values` is expected to be a dictionary"):
            d = {"id": "1", "values": [0.1, 0.2, 0.3], "metadata": {"genre": "comedy"}, "sparse_values": "not a dict"}
            VectorFactoryGRPC.build(d)

    def test_build_when_dict_sparse_values_errors_when_missing_indices(self):
        with pytest.raises(ValueError, match="Missing required keys in data in column `sparse_values`"):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "sparse_values": {"values": [0.1, 0.3]},
            }
            VectorFactoryGRPC.build(d)

    def test_build_when_dict_sparse_values_errors_when_missing_values(self):
        with pytest.raises(ValueError, match="Missing required keys in data in column `sparse_values`"):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "sparse_values": {"indices": [0, 2]},
            }
            VectorFactoryGRPC.build(d)

    def test_build_when_dict_sparse_values_errors_when_indices_not_list(self):
        with pytest.raises(ValueError, match="Found unexpected data in column `sparse_values`"):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "sparse_values": {"indices": "not a list", "values": [0.1, 0.3]},
            }
            VectorFactoryGRPC.build(d)

    def test_build_when_dict_sparse_values_errors_when_values_not_list(self):
        with pytest.raises(ValueError, match="Found unexpected data in column `sparse_values`"):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "sparse_values": {"indices": [0, 2], "values": "not a list"},
            }
            VectorFactoryGRPC.build(d)

    def test_build_when_dict_sparse_values_when_values_is_ndarray(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": {"indices": [0, 2], "values": np.array([0.1, 0.3])},
        }
        actual = VectorFactoryGRPC.build(d)
        expected = Vector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({"genre": "comedy"}),
            sparse_values=SparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    def test_build_when_dict_sparse_values_when_indices_is_pandas_IntegerArray(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": {"indices": pd.array([0, 2]), "values": [0.1, 0.3]},
        }
        actual = VectorFactoryGRPC.build(d)
        expected = Vector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({"genre": "comedy"}),
            sparse_values=SparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    def test_build_when_dict_sparse_values_when_values_is_pandas_FloatingArray(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": {"indices": [0, 2], "values": pd.array([0.1, 0.3])},
        }
        actual = VectorFactoryGRPC.build(d)
        expected = Vector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({"genre": "comedy"}),
            sparse_values=SparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    def test_build_when_dict_sparse_values_when_indices_is_ndarray(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": {"indices": np.array([0, 2]), "values": [0.1, 0.3]},
        }
        actual = VectorFactoryGRPC.build(d)
        expected = Vector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({"genre": "comedy"}),
            sparse_values=SparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    def test_build_when_errors_when_other_type(self):
        with pytest.raises(ValueError, match="Invalid vector value passed: cannot interpret type"):
            VectorFactoryGRPC.build(1)
