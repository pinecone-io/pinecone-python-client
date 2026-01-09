import numpy as np
import pandas as pd
import pytest

from pinecone.db_data.sparse_values_factory import SparseValuesFactory
from pinecone import SparseValues
from pinecone.core.openapi.db_data.models import SparseValues as OpenApiSparseValues
from pinecone.db_data.errors import (
    SparseValuesTypeError,
    SparseValuesMissingKeysError,
    SparseValuesDictionaryExpectedError,
)


class TestSparseValuesFactory:
    """Test SparseValuesFactory for REST API (db_data module)."""

    def test_build_when_none_returns_none(self):
        """Test that None input returns None."""
        assert SparseValuesFactory.build(None) is None

    def test_build_when_passed_openapi_sparse_values(self):
        """Test that OpenApiSparseValues are returned unchanged."""
        sv = OpenApiSparseValues(indices=[0, 2], values=[0.1, 0.3])
        actual = SparseValuesFactory.build(sv)
        assert actual == sv
        assert actual is sv

    def test_build_when_given_sparse_values_dataclass(self):
        """Test conversion from SparseValues dataclass to OpenApiSparseValues."""
        sv = SparseValues(indices=[0, 2], values=[0.1, 0.3])
        actual = SparseValuesFactory.build(sv)
        expected = OpenApiSparseValues(indices=[0, 2], values=[0.1, 0.3])
        assert isinstance(actual, OpenApiSparseValues)
        assert actual.indices == expected.indices
        assert actual.values == expected.values

    @pytest.mark.parametrize(
        "input_dict",
        [
            {"indices": [2], "values": [0.3]},
            {"indices": [88, 102], "values": [-0.1, 0.3]},
            {"indices": [0, 2, 4], "values": [0.1, 0.3, 0.5]},
            {"indices": [0, 2, 4, 6], "values": [0.1, 0.3, 0.5, 0.7]},
        ],
    )
    def test_build_when_valid_dictionary(self, input_dict):
        """Test building from valid dictionary input."""
        actual = SparseValuesFactory.build(input_dict)
        expected = OpenApiSparseValues(indices=input_dict["indices"], values=input_dict["values"])
        assert actual.indices == expected.indices
        assert actual.values == expected.values

    @pytest.mark.parametrize(
        "input_dict",
        [
            {"indices": np.array([0, 2]), "values": [0.1, 0.3]},
            {"indices": [0, 2], "values": np.array([0.1, 0.3])},
            {"indices": np.array([0, 2]), "values": np.array([0.1, 0.3])},
            {"indices": pd.array([0, 2]), "values": [0.1, 0.3]},
            {"indices": [0, 2], "values": pd.array([0.1, 0.3])},
            {"indices": pd.array([0, 2]), "values": pd.array([0.1, 0.3])},
        ],
    )
    def test_build_when_special_data_types(self, input_dict):
        """Test that the factory handles numpy/pandas arrays correctly."""
        actual = SparseValuesFactory.build(input_dict)
        expected = OpenApiSparseValues(indices=[0, 2], values=[0.1, 0.3])
        assert actual.indices == expected.indices
        assert actual.values == expected.values

    @pytest.mark.parametrize(
        "input_dict",
        [{"indices": [2], "values": [0.3, 0.3]}, {"indices": [88, 102], "values": [-0.1]}],
    )
    def test_build_when_list_sizes_dont_match(self, input_dict):
        """Test that mismatched indices and values lengths raise ValueError."""
        with pytest.raises(
            ValueError, match="Sparse values indices and values must have the same length"
        ):
            SparseValuesFactory.build(input_dict)

    @pytest.mark.parametrize(
        "input_dict",
        [
            {"indices": [2.0], "values": [0.3]},
            {"indices": ["2"], "values": [0.3]},
            {"indices": np.array([2.0]), "values": [0.3]},
            {"indices": pd.array([2.0]), "values": [0.3]},
        ],
    )
    def test_build_when_non_integer_indices(self, input_dict):
        """Test that non-integer indices raise SparseValuesTypeError."""
        with pytest.raises(SparseValuesTypeError):
            SparseValuesFactory.build(input_dict)

    @pytest.mark.parametrize(
        "input_dict", [{"indices": [2], "values": ["3.2"]}, {"indices": [2], "values": [True]}]
    )
    def test_build_when_non_float_values(self, input_dict):
        """Test that non-float values raise SparseValuesTypeError."""
        with pytest.raises(SparseValuesTypeError):
            SparseValuesFactory.build(input_dict)

    def test_build_when_missing_indices_key(self):
        """Test that missing 'indices' key raises SparseValuesMissingKeysError."""
        input_dict = {"values": [0.1, 0.3]}
        with pytest.raises(SparseValuesMissingKeysError) as exc_info:
            SparseValuesFactory.build(input_dict)
        assert "indices" in str(exc_info.value)

    def test_build_when_missing_values_key(self):
        """Test that missing 'values' key raises SparseValuesMissingKeysError."""
        input_dict = {"indices": [0, 2]}
        with pytest.raises(SparseValuesMissingKeysError) as exc_info:
            SparseValuesFactory.build(input_dict)
        assert "values" in str(exc_info.value)

    def test_build_when_missing_both_keys(self):
        """Test that missing both keys raises SparseValuesMissingKeysError."""
        input_dict = {}
        with pytest.raises(SparseValuesMissingKeysError) as exc_info:
            SparseValuesFactory.build(input_dict)
        assert "indices" in str(exc_info.value) or "values" in str(exc_info.value)

    def test_build_when_not_a_dictionary(self):
        """Test that non-dictionary input raises SparseValuesDictionaryExpectedError."""
        with pytest.raises(SparseValuesDictionaryExpectedError) as exc_info:
            SparseValuesFactory.build("not a dict")
        assert "dictionary" in str(exc_info.value).lower()

        with pytest.raises(SparseValuesDictionaryExpectedError):
            SparseValuesFactory.build(123)

        with pytest.raises(SparseValuesDictionaryExpectedError):
            SparseValuesFactory.build([1, 2, 3])

    def test_build_when_empty_indices_list(self):
        """Test that empty indices list is handled correctly."""
        input_dict = {"indices": [], "values": []}
        actual = SparseValuesFactory.build(input_dict)
        expected = OpenApiSparseValues(indices=[], values=[])
        assert actual.indices == expected.indices
        assert actual.values == expected.values

    def test_build_when_empty_values_list(self):
        """Test that empty values list is handled correctly."""
        input_dict = {"indices": [], "values": []}
        actual = SparseValuesFactory.build(input_dict)
        expected = OpenApiSparseValues(indices=[], values=[])
        assert actual.indices == expected.indices
        assert actual.values == expected.values
