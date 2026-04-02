"""Unit tests for VectorFactory in the db_data module."""

from array import array

import pytest

from pinecone.db_data.vector_factory import VectorFactory


class TestVectorFactoryTupleInput:
    """Tests for VectorFactory.build() with tuple input."""

    @pytest.mark.parametrize(
        "values",
        [
            [0.1, 0.2, 0.3],
            array("f", [0.1, 0.2, 0.3]),
        ],
    )
    def test_build_tuple_two_values(self, values):
        """VectorFactory accepts list[float] and array[float] in a two-element tuple."""
        result = VectorFactory.build(("id1", values))
        assert result.id == "id1"
        assert len(result.values) == 3

    @pytest.mark.parametrize(
        "values",
        [
            [0.1, 0.2, 0.3],
            array("f", [0.1, 0.2, 0.3]),
        ],
    )
    def test_build_tuple_three_values(self, values):
        """VectorFactory accepts list[float] and array[float] in a three-element tuple."""
        result = VectorFactory.build(("id1", values, {"genre": "comedy"}))
        assert result.id == "id1"
        assert len(result.values) == 3
        assert result.metadata == {"genre": "comedy"}
