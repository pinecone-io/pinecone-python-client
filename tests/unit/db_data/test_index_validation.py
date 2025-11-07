"""Unit tests for Index method parameter validation logic.

These tests replace integration tests that were making real API calls to test
client-side validation. They test the validation logic directly without
requiring API access.
"""

import pytest
from pinecone.db_data.vector_factory import VectorFactory


class TestIndexUpsertValidation:
    """Test parameter validation in Index.upsert()"""

    def test_vector_factory_validates_invalid_vector_types(self):
        """Test that VectorFactory validates vector types (replaces integration test)"""
        # This covers test_upsert_fails_when_vectors_wrong_type
        with pytest.raises(ValueError, match="Invalid vector value"):
            VectorFactory.build("not a vector")

        with pytest.raises(ValueError, match="Invalid vector value"):
            VectorFactory.build(123)

    def test_vector_factory_validates_missing_values(self):
        """Test that VectorFactory validates missing values (already covered by unit tests)"""
        # This is already tested in test_vector_factory.py
        # test_build_when_dict_missing_required_fields covers this
        with pytest.raises(ValueError, match="Vector dictionary is missing required fields"):
            VectorFactory.build({"values": [0.1, 0.2, 0.3]})  # Missing id

    def test_vector_factory_validates_missing_values_or_sparse_values(self):
        """Test that VectorFactory validates missing values/sparse_values (already covered by unit tests)"""
        # This is already tested in test_vector_factory.py
        # test_missing_values_and_sparse_values_dict covers this
        with pytest.raises(
            ValueError, match="At least one of 'values' or 'sparse_values' must be provided"
        ):
            VectorFactory.build({"id": "1"})  # Missing values and sparse_values
