import numpy as np
import pandas as pd
import pytest


from pinecone.grpc.vector_factory_grpc import VectorFactoryGRPC
from pinecone.grpc import GRPCVector, GRPCSparseValues
from pinecone.core.openapi.db_data.models import SparseValues as OpenApiSparseValues
from pinecone.grpc.utils import dict_to_proto_struct
from pinecone import Vector, SparseValues


class TestVectorFactoryGRPC_DenseVectors:
    def test_build_when_returns_vector_unmodified(self):
        vec = GRPCVector(id="1", values=[0.1, 0.2, 0.3])
        assert VectorFactoryGRPC.build(vec) == vec
        assert VectorFactoryGRPC.build(vec).__class__ == GRPCVector

    def test_build_converts_vector(self):
        vec = Vector(id="1", values=[0.1, 0.2, 0.3])
        assert VectorFactoryGRPC.build(vec).__class__ == GRPCVector
        assert VectorFactoryGRPC.build(vec) == GRPCVector(
            id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({})
        )

    def test_build_when_nongrpc_vector_it_converts(self):
        vec = Vector(id="1", values=[0.1, 0.2, 0.3])
        assert VectorFactoryGRPC.build(vec) == GRPCVector(
            id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({})
        )

    def test_build_when_nongrpc_vector_with_metadata_it_converts(self):
        vec = Vector(id="1", values=[0.1, 0.2, 0.3], metadata={"genre": "comedy"})
        assert VectorFactoryGRPC.build(vec) == GRPCVector(
            id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({"genre": "comedy"})
        )

    @pytest.mark.parametrize(
        "values_array", [[0.1, 0.2, 0.3], np.array([0.1, 0.2, 0.3]), pd.array([0.1, 0.2, 0.3])]
    )
    def test_build_when_tuple_with_two_values(self, values_array):
        tup = ("1", values_array)
        actual = VectorFactoryGRPC.build(tup)
        expected = GRPCVector(id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({}))
        assert actual == expected

    @pytest.mark.parametrize(
        "vector_tup",
        [
            ("1", "not an array"),
            ("1", {}),
            ("1", "not an array", {"genre": "comedy"}),
            ("1", {}, {"genre": "comedy"}),
        ],
    )
    def test_build_when_tuple_values_must_be_list(self, vector_tup):
        with pytest.raises(TypeError, match="Expected a list or list-like data structure"):
            VectorFactoryGRPC.build(vector_tup)

    @pytest.mark.parametrize(
        "values_array", [[0.1, 0.2, 0.3], np.array([0.1, 0.2, 0.3]), pd.array([0.1, 0.2, 0.3])]
    )
    def test_build_when_tuple_with_three_values(self, values_array):
        tup = ("1", values_array, {"genre": "comedy"})
        actual = VectorFactoryGRPC.build(tup)
        expected = GRPCVector(
            id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({"genre": "comedy"})
        )
        assert actual == expected

    def test_build_with_dict_no_metadata(self):
        d = {"id": "1", "values": [0.1, 0.2, 0.3]}
        actual = VectorFactoryGRPC.build(d)
        expected = GRPCVector(id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({}))
        assert actual == expected

    @pytest.mark.parametrize(
        "metadata", [{"genre": "comedy"}, dict_to_proto_struct({"genre": "comedy"})]
    )
    def test_build_when_dict_with_metadata2(self, metadata):
        d = {"id": "1", "values": [0.1, 0.2, 0.3], "metadata": metadata}
        actual = VectorFactoryGRPC.build(d)
        expected = GRPCVector(
            id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({"genre": "comedy"})
        )
        assert actual == expected

    def test_build_when_tuple_errors_when_additional_fields(self):
        with pytest.raises(ValueError, match="Found a tuple of length 4 which is not supported"):
            tup = ("1", [0.1, 0.2, 0.3], {"a": "b"}, "extra")
            VectorFactoryGRPC.build(tup)

    def test_build_when_tuple_too_short(self):
        with pytest.raises(ValueError, match="Found a tuple of length 1 which is not supported"):
            tup = ("1",)
            VectorFactoryGRPC.build(tup)

    @pytest.mark.parametrize("input_values", [pd.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3])])
    def test_build_when_dict_with_special_values(self, input_values):
        d = {"id": "1", "values": input_values, "metadata": {"genre": "comedy"}}
        actual = VectorFactoryGRPC.build(d)
        expected = GRPCVector(
            id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({"genre": "comedy"})
        )
        assert actual == expected

    @pytest.mark.parametrize("input_values", [pd.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3])])
    def test_build_when_Vector_with_special_values(self, input_values):
        d = Vector(**{"id": "1", "values": input_values, "metadata": {"genre": "comedy"}})
        actual = VectorFactoryGRPC.build(d)
        expected = GRPCVector(
            id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({"genre": "comedy"})
        )
        assert actual == expected

    @pytest.mark.parametrize("input_values", [pd.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3])])
    def test_build_when_Vector_with_special_sparse_values(self, input_values):
        d = Vector(
            id="1",
            values=input_values,
            sparse_values=SparseValues(indices=[0, 1, 2], values=input_values),
            metadata={"genre": "comedy"},
        )
        actual = VectorFactoryGRPC.build(d)
        expected = GRPCVector(
            id="1",
            values=[0.1, 0.2, 0.3],
            sparse_values=GRPCSparseValues(indices=[0, 1, 2], values=[0.1, 0.2, 0.3]),
            metadata=dict_to_proto_struct({"genre": "comedy"}),
        )
        assert actual == expected

    def test_build_when_dict_missing_required_fields(self):
        with pytest.raises(ValueError, match="Vector dictionary is missing required fields"):
            d = {"values": [0.1, 0.2, 0.3]}
            VectorFactoryGRPC.build(d)

    def test_build_when_dict_excess_keys(self):
        with pytest.raises(ValueError, match="Found excess keys in the vector dictionary"):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "extra": "field",
            }
            VectorFactoryGRPC.build(d)


class TestVectorFactoryGRPC_HybridVectors:
    def test_build_when_nongrpc_vector_with_sparse_values_it_converts(self):
        vec = Vector(
            id="1",
            values=[0.1, 0.2, 0.3],
            sparse_values=SparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert VectorFactoryGRPC.build(vec) == GRPCVector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({}),
            sparse_values=GRPCSparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )

    @pytest.mark.parametrize("sv_klass", [SparseValues, OpenApiSparseValues, GRPCSparseValues])
    def test_build_vector_with_tuple_with_sparse_values(self, sv_klass):
        tup = ("1", sv_klass(indices=[0, 2], values=[0.1, 0.3]))
        with pytest.raises(
            ValueError,
            match="Sparse values are not supported in tuples. Please use either dicts or Vector objects as inputs.",
        ):
            VectorFactoryGRPC.build(tup)

    def test_build_when_dict_with_sparse_values_dict(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": {"indices": [0, 2], "values": [0.1, 0.3]},
        }
        actual = VectorFactoryGRPC.build(d)
        expected = GRPCVector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({"genre": "comedy"}),
            sparse_values=GRPCSparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    @pytest.mark.parametrize("sv_klass", [SparseValues, OpenApiSparseValues, GRPCSparseValues])
    def test_build_with_dict_with_sparse_values_object(self, sv_klass):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": sv_klass(indices=[0, 2], values=[0.1, 0.3]),
        }
        actual = VectorFactoryGRPC.build(d)
        expected = GRPCVector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({"genre": "comedy"}),
            sparse_values=GRPCSparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    @pytest.mark.parametrize(
        "sv_indices,sv_values",
        [
            ([0, 2], [0.1, 0.3]),
            (pd.array([0, 2]), [0.1, 0.3]),
            ([0, 2], pd.array([0.1, 0.3])),
            (pd.array([0, 2]), pd.array([0.1, 0.3])),
            (np.array([0, 2]), [0.1, 0.3]),
            ([0, 2], np.array([0.1, 0.3])),
        ],
    )
    def test_build_when_dict_sparse_values(self, sv_indices, sv_values):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": {"indices": sv_indices, "values": sv_values},
        }
        actual = VectorFactoryGRPC.build(d)
        expected = GRPCVector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({"genre": "comedy"}),
            sparse_values=GRPCSparseValues(indices=[0, 2], values=[0.1, 0.3]),
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
        expected = GRPCVector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({"genre": "comedy"}),
            sparse_values=GRPCSparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    @pytest.mark.parametrize("bogus_sparse_values", [1, "not an array", [1, 2], {}])
    def test_build_when_dict_sparse_values_errors_when_invalid_sparse_values_values(
        self, bogus_sparse_values
    ):
        with pytest.raises(ValueError, match="Found unexpected data in column `sparse_values`"):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "sparse_values": {"indices": [1, 2], "values": bogus_sparse_values},
            }
            VectorFactoryGRPC.build(d)

    @pytest.mark.parametrize("bogus_sparse_indices", [1, "not an array", [0.1, 0.2], {}])
    def test_build_when_dict_sparse_values_errors_when_indices_not_valid_list(
        self, bogus_sparse_indices
    ):
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

    @pytest.mark.parametrize("bogus_sparse_values", [1, "not a dict", [1, 2, 3], []])
    def test_build_when_invalid_sparse_values_type_in_dict(self, bogus_sparse_values):
        with pytest.raises(
            ValueError, match="SparseValuesFactory does not know how to handle input type"
        ):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "sparse_values": bogus_sparse_values,  # not a valid dict
            }
            VectorFactoryGRPC.build(d)

    @pytest.mark.parametrize(
        "bogus_sparse_values", [{}, {"indices": [0, 2]}, {"values": [0.1, 0.3]}]
    )
    def test_build_when_missing_keys_in_sparse_values_dict(self, bogus_sparse_values):
        with pytest.raises(
            ValueError, match="Missing required keys in data in column `sparse_values`"
        ):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "sparse_values": bogus_sparse_values,
            }
            VectorFactoryGRPC.build(d)

    def test_build_when_sparse_values_is_None(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": None,
        }
        actual = VectorFactoryGRPC.build(d)
        expected = GRPCVector(
            id="1", values=[0.1, 0.2, 0.3], metadata=dict_to_proto_struct({"genre": "comedy"})
        )
        assert actual == expected

    @pytest.mark.parametrize(
        "sv",
        [
            SparseValues(indices=[0, 4], values=[0.1, 0.3]),
            GRPCSparseValues(indices=[0, 4], values=[0.1, 0.3]),
            OpenApiSparseValues(indices=[0, 4], values=[0.1, 0.3]),
        ],
    )
    def test_mixed_types_dict_with_sv_object(self, sv):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": sv,
        }
        actual = VectorFactoryGRPC.build(d)
        expected = GRPCVector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata=dict_to_proto_struct({"genre": "comedy"}),
            sparse_values=GRPCSparseValues(indices=[0, 4], values=[0.1, 0.3]),
        )
        assert actual == expected


class TestVectorFactoryGRPC_SparseVectors:
    def test_build_when_returns_vector_unmodified(self):
        vec = GRPCVector(
            id="1", sparse_values=GRPCSparseValues(indices=[0, 45, 11], values=[0.1, 0.2, 0.3])
        )
        assert VectorFactoryGRPC.build(vec) == vec
        assert VectorFactoryGRPC.build(vec).__class__ == GRPCVector

    def test_build_returns_grpc_vector(self):
        vec = Vector(
            id="1", sparse_values=SparseValues(indices=[0, 45, 11], values=[0.1, 0.2, 0.3])
        )
        assert VectorFactoryGRPC.build(vec).__class__ == GRPCVector
        assert VectorFactoryGRPC.build(vec) == GRPCVector(
            id="1",
            sparse_values=GRPCSparseValues(indices=[0, 45, 11], values=[0.1, 0.2, 0.3]),
            metadata=dict_to_proto_struct({}),
        )

    def test_no_error_when_empty_values_passed(self):
        v = Vector(id="1", values=[], sparse_values=SparseValues(indices=[0, 1], values=[0.1, 0.2]))
        actual = VectorFactoryGRPC.build(v)
        expected = GRPCVector(
            id="1",
            values=[],
            sparse_values=GRPCSparseValues(indices=[0, 1], values=[0.1, 0.2]),
            metadata=dict_to_proto_struct({}),
        )
        assert actual == expected

    @pytest.mark.parametrize("sv_klass", [SparseValues, GRPCSparseValues])
    def test_when_metadata(self, sv_klass):
        v = Vector(
            id="1",
            values=[],
            sparse_values=sv_klass(indices=[0, 1], values=[0.1, 0.2]),
            metadata={"genre": "comedy"},
        )
        actual = VectorFactoryGRPC.build(v)
        expected = GRPCVector(
            id="1",
            values=[],
            sparse_values=GRPCSparseValues(indices=[0, 1], values=[0.1, 0.2]),
            metadata=dict_to_proto_struct({"genre": "comedy"}),
        )
        assert actual == expected

    def test_when_dict_with_metadata(self):
        v = {
            "id": "1",
            "values": [],
            "sparse_values": {"indices": [0, 1], "values": [0.1, 0.2]},
            "metadata": {"genre": "comedy"},
        }
        actual = VectorFactoryGRPC.build(v)
        expected = GRPCVector(
            id="1",
            values=[],
            sparse_values=GRPCSparseValues(indices=[0, 1], values=[0.1, 0.2]),
            metadata=dict_to_proto_struct({"genre": "comedy"}),
        )
        assert actual == expected

    def test_build_when_dict(self):
        vec = {"id": "1", "sparse_values": {"indices": [0, 45, 11], "values": [0.1, 0.2, 0.3]}}
        actual = VectorFactoryGRPC.build(vec)
        expected = GRPCVector(
            id="1",
            sparse_values=GRPCSparseValues(indices=[0, 45, 11], values=[0.1, 0.2, 0.3]),
            metadata=dict_to_proto_struct({}),
        )
        assert actual == expected

    def test_build_when_dict_with_metadata3(self):
        vec = {
            "id": "1",
            "sparse_values": {"indices": [0, 45, 11], "values": [0.1, 0.2, 0.3]},
            "metadata": {"genre": "comedy"},
        }
        actual = VectorFactoryGRPC.build(vec)
        expected = GRPCVector(
            id="1",
            sparse_values=GRPCSparseValues(indices=[0, 45, 11], values=[0.1, 0.2, 0.3]),
            metadata=dict_to_proto_struct({"genre": "comedy"}),
        )
        assert actual == expected

    def test_build_when_mixed_types(self):
        vec = Vector(id="1", sparse_values={"indices": [0, 45, 11], "values": [0.1, 0.2, 0.3]})
        actual = VectorFactoryGRPC.build(vec)
        expected = GRPCVector(
            id="1",
            sparse_values=GRPCSparseValues(indices=[0, 45, 11], values=[0.1, 0.2, 0.3]),
            metadata=dict_to_proto_struct({}),
        )
        assert actual == expected

    def test_build_when_nongrpc_vector_it_converts(self):
        vec = Vector(
            id="1", sparse_values=SparseValues(indices=[0, 45, 11], values=[0.1, 0.2, 0.3])
        )
        assert VectorFactoryGRPC.build(vec) == GRPCVector(
            id="1",
            sparse_values=GRPCSparseValues(indices=[0, 45, 11], values=[0.1, 0.2, 0.3]),
            metadata=dict_to_proto_struct({}),
        )

    def test_build_when_nongrpc_vector_with_metadata_it_converts(self):
        vec = Vector(
            id="1",
            sparse_values=SparseValues(indices=[0, 45, 11], values=[0.1, 0.2, 0.3]),
            metadata={"genre": "comedy"},
        )
        assert VectorFactoryGRPC.build(vec) == GRPCVector(
            id="1",
            sparse_values=GRPCSparseValues(indices=[0, 45, 11], values=[0.1, 0.2, 0.3]),
            metadata=dict_to_proto_struct({"genre": "comedy"}),
        )

    def test_build_with_dict_no_metadata(self):
        d = {"id": "1", "sparse_values": {"indices": [23, 13, 45], "values": [0.1, 0.2, 0.3]}}
        actual = VectorFactoryGRPC.build(d)
        expected = GRPCVector(
            id="1",
            sparse_values=GRPCSparseValues(indices=[23, 13, 45], values=[0.1, 0.2, 0.3]),
            metadata=dict_to_proto_struct({}),
        )
        assert actual == expected

    @pytest.mark.parametrize(
        "metadata", [{"genre": "comedy"}, dict_to_proto_struct({"genre": "comedy"})]
    )
    def test_build_when_dict_with_metadata(self, metadata):
        d = {
            "id": "1",
            "sparse_values": {"indices": [23, 13, 45], "values": [0.1, 0.2, 0.3]},
            "metadata": metadata,
        }
        actual = VectorFactoryGRPC.build(d)
        expected = GRPCVector(
            id="1",
            sparse_values=GRPCSparseValues(indices=[23, 13, 45], values=[0.1, 0.2, 0.3]),
            metadata=dict_to_proto_struct({"genre": "comedy"}),
        )
        assert actual == expected

    @pytest.mark.parametrize("sv_values", [pd.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3])])
    def test_build_when_dict_with_special_values(self, sv_values):
        d = {
            "id": "1",
            "sparse_values": {"indices": [10, 12, 14], "values": sv_values},
            "metadata": {"genre": "comedy"},
        }
        actual = VectorFactoryGRPC.build(d)
        expected = GRPCVector(
            id="1",
            sparse_values=GRPCSparseValues(values=[0.1, 0.2, 0.3], indices=[10, 12, 14]),
            metadata=dict_to_proto_struct({"genre": "comedy"}),
        )
        assert actual == expected

    @pytest.mark.parametrize("sv_indices", [pd.array([1, 2, 3]), np.array([1, 2, 3])])
    def test_build_when_dict_with_special_indices(self, sv_indices):
        d = {
            "id": "1",
            "sparse_values": {"indices": sv_indices, "values": [0.1, 0.2, 0.3]},
            "metadata": {"genre": "comedy"},
        }
        actual = VectorFactoryGRPC.build(d)
        expected = GRPCVector(
            id="1",
            sparse_values=GRPCSparseValues(values=[0.1, 0.2, 0.3], indices=[1, 2, 3]),
            metadata=dict_to_proto_struct({"genre": "comedy"}),
        )
        assert actual == expected

    def test_build_when_dict_missing_required_fields(self):
        with pytest.raises(ValueError) as e:
            d = {"sparse_values": {"values": [0.1, 0.2, 0.3], "indices": [1, 2, 3]}}
            VectorFactoryGRPC.build(d)

        assert "Vector dictionary is missing required fields" in str(e)
        assert "id" in str(e)

    def test_spare_values_missing_indices(self):
        with pytest.raises(ValueError) as e:
            d = {"id": "1", "sparse_values": {"values": [0.1, 0.2, 0.3]}}
            VectorFactoryGRPC.build(d)

        assert "Missing required keys" in str(e)
        assert "indices" in str(e)

    def test_build_when_dict_excess_keys(self):
        with pytest.raises(ValueError, match="Found excess keys in the vector dictionary"):
            d = {
                "id": "1",
                "sparse_values": {"values": [0.1, 0.2, 0.3], "indices": [1, 2, 3]},
                "metadata": {"genre": "comedy"},
                "extra": "field",
            }
            VectorFactoryGRPC.build(d)


class TestVectorFactoryGRPC_EdgeCases:
    def test_must_have_values_or_sparse_values(self):
        with pytest.raises(
            ValueError,
            match="At least one of 'values' or 'sparse_values' must be provided in the vector dictionary.",
        ):
            VectorFactoryGRPC.build({"id": "1"})
