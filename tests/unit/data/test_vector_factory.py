import numpy as np
import pandas as pd
import pytest

from pinecone.db_data.vector_factory import VectorFactory
from pinecone import Vector, SparseValues, ListConversionException
from pinecone.core.openapi.db_data.models import (
    Vector as OpenApiVector,
    SparseValues as OpenApiSparseValues,
)


class TestVectorFactory_DenseVectors:
    def test_build_when_returns_vector_unmodified(self):
        vec = Vector(id="1", values=[0.1, 0.2, 0.3])
        assert VectorFactory.build(vec) == OpenApiVector(id="1", values=[0.1, 0.2, 0.3])
        assert VectorFactory.build(vec).__class__ == OpenApiVector

    @pytest.mark.parametrize(
        "values_array", [[0.1, 0.2, 0.3], np.array([0.1, 0.2, 0.3]), pd.array([0.1, 0.2, 0.3])]
    )
    def test_build_when_tuple_with_two_values(self, values_array):
        tup = ("1", values_array)
        actual = VectorFactory.build(tup)
        expected = OpenApiVector(id="1", values=[0.1, 0.2, 0.3], metadata={})
        assert actual == expected

    @pytest.mark.parametrize(
        "values_array", [[0.1, 0.2, 0.3], np.array([0.1, 0.2, 0.3]), pd.array([0.1, 0.2, 0.3])]
    )
    def test_build_when_tuple_with_three_values(self, values_array):
        tup = ("1", values_array, {"genre": "comedy"})
        actual = VectorFactory.build(tup)
        expected = OpenApiVector(id="1", values=[0.1, 0.2, 0.3], metadata={"genre": "comedy"})
        assert actual == expected

    @pytest.mark.parametrize(
        "vector_tup",
        [
            ("1", "not an array"),
            ("1", {}),
            ("1", None),
            ("1", "not an array", {"genre": "comedy"}),
            ("1", {}, {"genre": "comedy"}),
            ("1", None, {"genre": "comedy"}),
        ],
    )
    def test_build_when_tuple_values_must_be_list(self, vector_tup):
        with pytest.raises(
            ListConversionException, match="Expected a list or list-like data structure"
        ):
            VectorFactory.build(vector_tup)

    def test_build_when_tuple_errors_when_additional_fields(self):
        with pytest.raises(ValueError, match="Found a tuple of length 4 which is not supported"):
            tup = ("1", [0.1, 0.2, 0.3], {"a": "b"}, "extra")
            VectorFactory.build(tup)

    def test_build_when_tuple_too_short(self):
        with pytest.raises(ValueError, match="Found a tuple of length 1 which is not supported"):
            tup = ("1",)
            VectorFactory.build(tup)

    @pytest.mark.parametrize(
        "values_array", [[0.1, 0.2, 0.3], np.array([0.1, 0.2, 0.3]), pd.array([0.1, 0.2, 0.3])]
    )
    def test_build_when_dict(self, values_array):
        d = {"id": "1", "values": values_array, "metadata": {"genre": "comedy"}}
        actual = VectorFactory.build(d)
        expected = OpenApiVector(id="1", values=[0.1, 0.2, 0.3], metadata={"genre": "comedy"})
        assert actual == expected

    def test_build_when_dict_missing_required_fields(self):
        with pytest.raises(ValueError, match="Vector dictionary is missing required fields"):
            d = {"values": [0.1, 0.2, 0.3]}
            VectorFactory.build(d)

    def test_build_when_dict_excess_keys(self):
        with pytest.raises(ValueError, match="Found excess keys in the vector dictionary"):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "extra": "field",
            }
            VectorFactory.build(d)


class TestVectorFactory_HybridVectors:
    def test_build_when_dict_sparse_values(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": {"indices": [0, 2], "values": [0.1, 0.3]},
        }
        actual = VectorFactory.build(d)
        expected = OpenApiVector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata={"genre": "comedy"},
            sparse_values=OpenApiSparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    def test_build_when_dict_sparse_values_when_SparseValues(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": SparseValues(indices=[0, 2], values=[0.1, 0.3]),
        }
        actual = VectorFactory.build(d)
        expected = OpenApiVector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata={"genre": "comedy"},
            sparse_values=OpenApiSparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    def test_build_when_dict_sparse_values_errors_when_not_dict(self):
        with pytest.raises(
            ValueError, match="Column `sparse_values` is expected to be a dictionary"
        ):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "sparse_values": "not a dict",
            }
            VectorFactory.build(d)

    def test_build_when_dict_sparse_values_errors_when_missing_indices(self):
        with pytest.raises(
            ValueError, match="Missing required keys in data in column `sparse_values`"
        ):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "sparse_values": {"values": [0.1, 0.3]},
            }
            VectorFactory.build(d)

    def test_build_when_dict_sparse_values_errors_when_missing_values(self):
        with pytest.raises(
            ValueError, match="Missing required keys in data in column `sparse_values`"
        ):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "sparse_values": {"indices": [0, 2]},
            }
            VectorFactory.build(d)

    def test_build_when_dict_sparse_values_errors_when_indices_not_list(self):
        with pytest.raises(ValueError, match="Found unexpected data in column `sparse_values`"):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "sparse_values": {"indices": "not a list", "values": [0.1, 0.3]},
            }
            VectorFactory.build(d)

    def test_build_when_dict_sparse_values_errors_when_values_not_list(self):
        with pytest.raises(ValueError, match="Found unexpected data in column `sparse_values`"):
            d = {
                "id": "1",
                "values": [0.1, 0.2, 0.3],
                "metadata": {"genre": "comedy"},
                "sparse_values": {"indices": [0, 2], "values": "not a list"},
            }
            VectorFactory.build(d)

    def test_build_when_dict_sparse_values_when_values_is_ndarray(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": {"indices": [0, 2], "values": np.array([0.1, 0.3])},
        }
        actual = VectorFactory.build(d)
        expected = OpenApiVector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata={"genre": "comedy"},
            sparse_values=OpenApiSparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    def test_build_when_dict_sparse_values_when_indices_is_pandas_IntegerArray(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": {"indices": pd.array([0, 2]), "values": [0.1, 0.3]},
        }
        actual = VectorFactory.build(d)
        expected = OpenApiVector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata={"genre": "comedy"},
            sparse_values=OpenApiSparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    def test_build_when_dict_sparse_values_when_values_is_pandas_FloatingArray(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": {"indices": [0, 2], "values": pd.array([0.1, 0.3])},
        }
        actual = VectorFactory.build(d)
        expected = OpenApiVector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata={"genre": "comedy"},
            sparse_values=OpenApiSparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    def test_build_when_dict_sparse_values_when_indices_is_ndarray(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": {"indices": np.array([0, 2]), "values": [0.1, 0.3]},
        }
        actual = VectorFactory.build(d)
        expected = OpenApiVector(
            id="1",
            values=[0.1, 0.2, 0.3],
            metadata={"genre": "comedy"},
            sparse_values=OpenApiSparseValues(indices=[0, 2], values=[0.1, 0.3]),
        )
        assert actual == expected

    def test_build_when_errors_when_other_type(self):
        with pytest.raises(ValueError, match="Invalid vector value passed: cannot interpret type"):
            VectorFactory.build(1)

    def test_build_when_sparse_values_is_None(self):
        d = {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"genre": "comedy"},
            "sparse_values": None,
        }
        actual = VectorFactory.build(d)
        expected = OpenApiVector(id="1", values=[0.1, 0.2, 0.3], metadata={"genre": "comedy"})
        assert actual == expected


class TestVectorFactory_SparseVectors:
    def test_when_sparse_only_it_sets_empty_values_obj(self):
        o = Vector(id="1", sparse_values=SparseValues(indices=[0, 2], values=[0.1, 0.3]))
        actual = VectorFactory.build(o)
        expected = OpenApiVector(
            id="1", values=[], sparse_values=OpenApiSparseValues(indices=[0, 2], values=[0.1, 0.3])
        )
        assert actual == expected

    def test_sparse_when_dict(self):
        d = {"id": "1", "sparse_values": {"indices": [0, 2], "values": [0.1, 0.3]}}
        actual = VectorFactory.build(d)
        expected = OpenApiVector(
            id="1", values=[], sparse_values=OpenApiSparseValues(indices=[0, 2], values=[0.1, 0.3])
        )
        assert actual == expected

    def test_sparse_when_dict_with_object(self):
        d = {"id": "1", "sparse_values": SparseValues(indices=[0, 2], values=[0.1, 0.3])}
        actual = VectorFactory.build(d)
        expected = OpenApiVector(
            id="1", values=[], sparse_values=OpenApiSparseValues(indices=[0, 2], values=[0.1, 0.3])
        )
        assert actual == expected

    def test_sparse_when_object_with_dict(self):
        o = Vector(id="1", sparse_values={"indices": [0, 2], "values": [0.1, 0.3]})
        actual = VectorFactory.build(o)
        expected = OpenApiVector(
            id="1", values=[], sparse_values=OpenApiSparseValues(indices=[0, 2], values=[0.1, 0.3])
        )
        assert actual == expected

    def test_when_passing_openapi_objects_should_be_unmodified(self):
        """Not really expecting this, but it's possible somebody could do it."""
        o = OpenApiVector(id="1", values=[0.1, 0.2, 0.3])
        actual = VectorFactory.build(o)
        expected = OpenApiVector(id="1", values=[0.1, 0.2, 0.3])
        assert actual == expected

    def test_sparse_dict_when_missing_keys(self):
        with pytest.raises(
            ValueError, match="Missing required keys in data in column `sparse_values`"
        ):
            d = {"id": "1", "sparse_values": {}}
            VectorFactory.build(d)


class TestVectorFactory_EdgeCases:
    def test_missing_values_and_sparse_values_dict(self):
        with pytest.raises(
            ValueError,
            match="At least one of 'values' or 'sparse_values' must be provided in the vector dictionary.",
        ):
            VectorFactory.build({"id": "1"})

    def test_missing_values_and_sparse_values_obj(self):
        with pytest.raises(
            ValueError, match="The values and sparse_values fields cannot both be empty"
        ):
            VectorFactory.build(Vector(id="1"))

    def test_missing_id(self):
        with pytest.raises(ValueError, match="Vector dictionary is missing required fields"):
            VectorFactory.build({"values": [0.1, 0.2, 0.3]})
        with pytest.raises(ValueError, match="Vector dictionary is missing required fields"):
            VectorFactory.build({"sparse_values": {"indices": [0, 2], "values": [0.1, 0.3]}})
