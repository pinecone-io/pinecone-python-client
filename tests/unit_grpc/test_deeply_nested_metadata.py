"""Regression tests for CVE-2025-4565.

Tests handling of deeply nested protobuf structures to prevent RecursionError
when parsing untrusted metadata with recursive groups/messages.
"""

from pinecone.grpc.utils import dict_to_proto_struct, _struct_to_dict


def test_deeply_nested_protobuf_metadata():
    """Regression test for CVE-2025-4565: Handle deeply nested metadata without RecursionError.

    Tests that deeply nested dictionary structures can be converted to protobuf Structs
    and back without hitting Python's recursion limit.
    """
    # Create metadata with significant nesting depth
    nested_dict = {}
    current = nested_dict
    for i in range(50):
        current["nested"] = {}
        current = current["nested"]
    current["value"] = "deep"

    # Should parse without RecursionError
    struct = dict_to_proto_struct(nested_dict)
    result = _struct_to_dict(struct)
    assert result is not None


def test_nested_lists_in_metadata():
    """Test nested lists don't cause recursion issues.

    Verifies that deeply nested list structures in metadata can be handled
    without triggering recursion errors.
    """
    metadata = {"list": [[[[{"key": "value"}]]]]}
    struct = dict_to_proto_struct(metadata)
    result = _struct_to_dict(struct)
    assert result is not None


def test_mixed_nested_structures():
    """Test mixed nested dictionaries and lists.

    Ensures that complex combinations of nested dictionaries and lists
    can be processed safely.
    """
    metadata = {
        "level1": {
            "level2": [{"level3": {"level4": [{"level5": "value"}]}}, {"another": [1, 2, 3]}]
        }
    }
    struct = dict_to_proto_struct(metadata)
    result = _struct_to_dict(struct)
    assert result is not None
    assert result["level1"]["level2"][0]["level3"]["level4"][0]["level5"] == "value"
