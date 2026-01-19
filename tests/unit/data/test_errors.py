from pinecone.db_data.errors import (
    VectorDictionaryMissingKeysError,
    VectorDictionaryExcessKeysError,
    VectorTupleLengthError,
    SparseValuesTypeError,
    SparseValuesMissingKeysError,
    SparseValuesDictionaryExpectedError,
    MetadataDictionaryExpectedError,
)


class TestVectorDictionaryMissingKeysError:
    """Test VectorDictionaryMissingKeysError exception."""

    def test_error_message_includes_missing_keys(self):
        """Test that error message lists missing required fields."""
        item = {"values": [0.1, 0.2]}
        error = VectorDictionaryMissingKeysError(item)
        assert "missing required fields" in str(error).lower()
        assert "id" in str(error)

    def test_error_message_with_multiple_missing_keys(self):
        """Test error message when multiple keys are missing."""
        item = {}
        error = VectorDictionaryMissingKeysError(item)
        assert "missing required fields" in str(error).lower()


class TestVectorDictionaryExcessKeysError:
    """Test VectorDictionaryExcessKeysError exception."""

    def test_error_message_includes_excess_keys(self):
        """Test that error message lists excess keys."""
        item = {"id": "1", "values": [0.1, 0.2], "extra_field": "value", "another_extra": 123}
        error = VectorDictionaryExcessKeysError(item)
        assert "excess keys" in str(error).lower()
        assert "extra_field" in str(error) or "another_extra" in str(error)

    def test_error_message_includes_allowed_keys(self):
        """Test that error message includes list of allowed keys."""
        item = {"id": "1", "values": [0.1, 0.2], "invalid": "key"}
        error = VectorDictionaryExcessKeysError(item)
        assert "allowed keys" in str(error).lower()


class TestVectorTupleLengthError:
    """Test VectorTupleLengthError exception."""

    def test_error_message_includes_tuple_length(self):
        """Test that error message includes the tuple length."""
        item = ("id", "values", "metadata", "extra")
        error = VectorTupleLengthError(item)
        assert str(len(item)) in str(error)
        assert "tuple" in str(error).lower()

    def test_error_message_with_length_one(self):
        """Test error message for tuple of length 1."""
        item = ("id",)
        error = VectorTupleLengthError(item)
        assert "1" in str(error)

    def test_error_message_with_length_four(self):
        """Test error message for tuple of length 4."""
        item = ("id", "values", "metadata", "extra")
        error = VectorTupleLengthError(item)
        assert "4" in str(error)


class TestSparseValuesTypeError:
    """Test SparseValuesTypeError exception."""

    def test_error_message_mentions_sparse_values(self):
        """Test that error message mentions sparse_values."""
        error = SparseValuesTypeError()
        assert "sparse_values" in str(error).lower()

    def test_error_is_both_value_and_type_error(self):
        """Test that SparseValuesTypeError is both ValueError and TypeError."""
        error = SparseValuesTypeError()
        assert isinstance(error, ValueError)
        assert isinstance(error, TypeError)


class TestSparseValuesMissingKeysError:
    """Test SparseValuesMissingKeysError exception."""

    def test_error_message_includes_found_keys(self):
        """Test that error message includes the keys that were found."""
        sparse_values_dict = {"indices": [0, 2]}
        error = SparseValuesMissingKeysError(sparse_values_dict)
        assert "missing required keys" in str(error).lower()
        assert "indices" in str(error) or "values" in str(error)

    def test_error_message_with_empty_dict(self):
        """Test error message when dictionary is empty."""
        sparse_values_dict = {}
        error = SparseValuesMissingKeysError(sparse_values_dict)
        assert "missing required keys" in str(error).lower()


class TestSparseValuesDictionaryExpectedError:
    """Test SparseValuesDictionaryExpectedError exception."""

    def test_error_message_includes_actual_type(self):
        """Test that error message includes the actual type found."""
        sparse_values_dict = "not a dict"
        error = SparseValuesDictionaryExpectedError(sparse_values_dict)
        assert "dictionary" in str(error).lower()
        assert "str" in str(error) or type(sparse_values_dict).__name__ in str(error)

    def test_error_message_with_integer(self):
        """Test error message when integer is provided."""
        sparse_values_dict = 123
        error = SparseValuesDictionaryExpectedError(sparse_values_dict)
        assert "dictionary" in str(error).lower()
        assert isinstance(error, ValueError)
        assert isinstance(error, TypeError)


class TestMetadataDictionaryExpectedError:
    """Test MetadataDictionaryExpectedError exception."""

    def test_error_message_includes_actual_type(self):
        """Test that error message includes the actual type found."""
        item = {"metadata": "not a dict"}
        error = MetadataDictionaryExpectedError(item)
        assert "dictionary" in str(error).lower()
        assert "metadata" in str(error).lower()

    def test_error_message_with_list(self):
        """Test error message when list is provided as metadata."""
        item = {"metadata": [1, 2, 3]}
        error = MetadataDictionaryExpectedError(item)
        assert "dictionary" in str(error).lower()
        assert isinstance(error, ValueError)
        assert isinstance(error, TypeError)
