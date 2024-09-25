import pytest
from pinecone.grpc import SparseValues as GRPCSparseValues
from pinecone import SparseValues as NonGRPCSparseValues

import numpy as np
import pandas as pd


from pinecone.grpc.sparse_values_factory import SparseValuesFactory


class TestSparseValuesFactory:
    def test_build_when_None(self):
        assert SparseValuesFactory.build(None) is None

    def test_build_when_passed_GRPCSparseValues(self):
        """
        Return without modification when given GRPCSparseValues
        """
        sv = GRPCSparseValues(indices=[0, 2], values=[0.1, 0.3])
        actual = SparseValuesFactory.build(sv)
        assert actual == sv

    def test_build_when_passed_NonGRPCSparseValues(self):
        """
        Convert when given NonGRPCSparseValues
        """
        sv = NonGRPCSparseValues(indices=[0, 2], values=[0.1, 0.3])
        actual = SparseValuesFactory.build(sv)
        expected = GRPCSparseValues(indices=[0, 2], values=[0.1, 0.3])
        assert actual == expected

    @pytest.mark.parametrize(
        "input",
        [
            {"indices": [2], "values": [0.3]},
            {"indices": [88, 102], "values": [-0.1, 0.3]},
            {"indices": [0, 2, 4], "values": [0.1, 0.3, 0.5]},
            {"indices": [0, 2, 4, 6], "values": [0.1, 0.3, 0.5, 0.7]},
        ],
    )
    def test_build_when_valid_dictionary(self, input):
        actual = SparseValuesFactory.build(input)
        expected = GRPCSparseValues(indices=input["indices"], values=input["values"])
        assert actual == expected

    @pytest.mark.parametrize(
        "input",
        [
            {"indices": np.array([0, 2]), "values": [0.1, 0.3]},
            {"indices": [0, 2], "values": np.array([0.1, 0.3])},
            {"indices": np.array([0, 2]), "values": np.array([0.1, 0.3])},
            {"indices": pd.array([0, 2]), "values": [0.1, 0.3]},
            {"indices": [0, 2], "values": pd.array([0.1, 0.3])},
            {"indices": pd.array([0, 2]), "values": pd.array([0.1, 0.3])},
            {"indices": np.array([0, 2]), "values": pd.array([0.1, 0.3])},
            {"indices": pd.array([0, 2]), "values": np.array([0.1, 0.3])},
        ],
    )
    def test_build_when_special_data_types(self, input):
        """
        Test that the factory can handle special data types like
        numpy/pandas integer and float arrays.
        """
        actual = SparseValuesFactory.build(input)
        expected = GRPCSparseValues(indices=[0, 2], values=[0.1, 0.3])
        assert actual == expected

    @pytest.mark.parametrize(
        "input", [{"indices": [2], "values": [0.3, 0.3]}, {"indices": [88, 102], "values": [-0.1]}]
    )
    def test_build_when_list_sizes_dont_match(self, input):
        with pytest.raises(
            ValueError, match="Sparse values indices and values must have the same length"
        ):
            SparseValuesFactory.build(input)

    @pytest.mark.parametrize(
        "input",
        [
            {"indices": [2.0], "values": [0.3]},
            {"indices": ["2"], "values": [0.3]},
            {"indices": np.array([2.0]), "values": [0.3]},
            {"indices": pd.array([2.0]), "values": [0.3]},
        ],
    )
    def test_build_when_non_integer_indices(self, input):
        with pytest.raises(ValueError, match="Found unexpected data in column `sparse_values`"):
            SparseValuesFactory.build(input)

    @pytest.mark.parametrize(
        "input",
        [
            {"indices": [2], "values": [3]},
            {"indices": [2], "values": ["3.2"]},
            {"indices": [2], "values": np.array([3])},
            {"indices": [2], "values": pd.array([3])},
        ],
    )
    def test_build_when_non_float_values(self, input):
        with pytest.raises(ValueError, match="Found unexpected data in column `sparse_values`"):
            SparseValuesFactory.build(input)
