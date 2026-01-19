from pinecone.utils.fix_tuple_length import fix_tuple_length


class TestFixTupleLength:
    """Test fix_tuple_length utility function."""

    def test_tuple_shorter_than_target(self):
        """Test extending a tuple that's shorter than target length."""
        result = fix_tuple_length(("a", "b"), 4)
        assert result == ("a", "b", None, None)
        assert len(result) == 4

    def test_tuple_equal_to_target(self):
        """Test tuple that's already the target length."""
        result = fix_tuple_length(("a", "b", "c"), 3)
        assert result == ("a", "b", "c")
        assert len(result) == 3

    def test_tuple_longer_than_target(self):
        """Test tuple that's longer than target length (should be unchanged)."""
        result = fix_tuple_length(("a", "b", "c", "d"), 3)
        assert result == ("a", "b", "c", "d")
        assert len(result) == 4

    def test_empty_tuple_extended(self):
        """Test extending an empty tuple."""
        result = fix_tuple_length((), 3)
        assert result == (None, None, None)
        assert len(result) == 3

    def test_single_element_tuple(self):
        """Test extending a single-element tuple."""
        result = fix_tuple_length(("a",), 3)
        assert result == ("a", None, None)
        assert len(result) == 3

    def test_extend_to_length_one(self):
        """Test extending to length 1."""
        result = fix_tuple_length((), 1)
        assert result == (None,)
        assert len(result) == 1

    def test_extend_to_length_zero(self):
        """Test extending to length 0 (edge case)."""
        result = fix_tuple_length((), 0)
        assert result == ()
        assert len(result) == 0

    def test_preserves_original_values(self):
        """Test that original values are preserved in correct positions."""
        result = fix_tuple_length(("id", "values", "metadata"), 5)
        assert result[0] == "id"
        assert result[1] == "values"
        assert result[2] == "metadata"
        assert result[3] is None
        assert result[4] is None

    def test_with_none_values(self):
        """Test tuple that already contains None values."""
        result = fix_tuple_length(("a", None, "c"), 5)
        assert result == ("a", None, "c", None, None)
        assert result[1] is None
        assert result[3] is None
        assert result[4] is None
