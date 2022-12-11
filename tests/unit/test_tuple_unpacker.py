import pytest
import re
from typing import List, Dict, Any

from core.utils.tuple_unpacker import TupleUnpacker


class TestTupleUnpacker:

    def test_unpack_OnlyRequieredFieldsHappyPath_ReturnInputTuple(self):
        tup = ("id", 3, [3.0, 4.5], {1: 2.0, 3: 4.0}, {"a": 1.0, "b": 2.0})
        ordered_required_items = [('id', str),
                                  ('int', int),
                                  ('vector', List[float]),
                                  ('sparse_vector', Dict[int, float]),
                                  ('metadata', Dict[str, Any])]

        assert TupleUnpacker.unpack(tup, ordered_required_items, []) == tup

    def test_unpack_OnlyRequieredFieldsMissingRequiredField_ThrowException(self):
        tup = ("id", 3, [3.0, 4.5], {1: 2.0, 3: 4.0})
        ordered_required_items = [('id', str),
                                  ('int', int),
                                  ('vector', List[float]),
                                  ('sparse_vector', Dict[int, float]),
                                  ('metadata', Dict[str, Any])]

        with pytest.raises(TypeError,
                           match=re.escape(
                               "Missing required arguments in input tuple.\n"
                               "Argument '('metadata', typing.Dict[str, typing.Any])' is missing at position 4.\n"
                               "Input tuple: ('id', 3, [3.0, 4.5], {1: 2.0, 3: 4.0})")):

            TupleUnpacker.unpack(tup, ordered_required_items, [])

    def test_unpack_OnlyRequieredFieldsWrongType_ThrowException(self):
        ordered_required_items = [('id', str),
                                  ('int', int),
                                  ('vector', List[float]),
                                  ('sparse_vector', Dict[int, float]),
                                  ('metadata', Dict[str, Any])]

        tup = (7, 3, [3.0, 4.5], {1: 2.0, 3: 4.0}, {"a": 1.0, "b": 2.0})
        with pytest.raises(TypeError,
                           match=r"Argument 'id' in position 0 must be of type: <class 'str'>.\nReceived value: 7"):

            TupleUnpacker.unpack(tup, ordered_required_items, [])

        tup = ("id", 3, [3, 4], {1: 2.0, 3: 4.0}, {"a": 1.0, "b": 2.0})
        with pytest.raises(TypeError,
                           match=re.escape("Argument 'vector' in position 2 must be of type: typing.List[float]."
                                           "\nReceived value: [3, 4]")):

            TupleUnpacker.unpack(tup, ordered_required_items, [])

        tup = ("id", 3, [3.0, 4.5], {"1": 2.0, 3: 4.0}, {"a": 1, "b": 2.0})
        with pytest.raises(
                TypeError,
                match=re.escape("Argument 'sparse_vector' in position 3 must be of type: typing.Dict[int, float]."
                                "\nReceived value: {'1': 2.0, 3: 4.0}")):

            TupleUnpacker.unpack(tup, ordered_required_items, [])

    def test_unpack_RequiredAndOptionalAllOptionalInInput_ReturnInputTuple(self):
        tup = ("id", 3, [3.0, 4.5], {1: 2.0, 3: 4.0}, {"a": 1.0, "b": 2.0})
        ordered_required_items = [('id', str),
                                  ('int', int),
                                  ('vector', List[float])]
        ordered_optional_items = [('sparse_vector', Dict[int, float]), ('metadata', Dict[str, Any])]

        assert TupleUnpacker.unpack(tup, ordered_required_items, ordered_optional_items) == tup

    def test_unpack_RequiredAndOptionalNoOptionalInInput_ReturnInputTupleWithNones(self):
        tup = ("id", 3, [3.0, 4.5])
        ordered_required_items = [('id', str),
                                  ('int', int),
                                  ('vector', List[float])]
        ordered_optional_items = [('sparse_vector', Dict[int, float]), ('metadata', Dict[str, Any])]

        assert TupleUnpacker.unpack(tup, ordered_required_items, ordered_optional_items) == tup + (None, None)

    def test_unpackRequiredAndOptionalAllOptionalInInputWrongType_ThrowException(self):
        tup = ("id", 3, [3.0, 4.5], {1: 2.0, 3: 4.0}, {1: 1, "b": 2.0})
        ordered_required_items = [('id', str),
                                  ('int', int),
                                  ('vector', List[float])]
        ordered_optional_items = [('sparse_vector', Dict[int, float]), ('metadata', Dict[str, Any])]

        with pytest.raises(
                TypeError,
                match=re.escape("Unexpected argument in position 4.\n"
                                "Input tuple: ('id', 3, [3.0, 4.5], {1: 2.0, 3: 4.0}, {1: 1, 'b': 2.0})")):

            TupleUnpacker.unpack(tup, ordered_required_items, ordered_optional_items)

    def test_unpack_onlySomeOfOptionalInInput_FillNoneOnMissingPostions(self):
        ordered_required_items = [('id', str),
                                  ('int', int),
                                  ('vector', List[float])]
        ordered_optional_items = [('sparse_vector', Dict[int, float]), ('metadata', Dict[str, Any])]

        tup = ("id", 3, [3.0, 4.5], {1: 2.0, 3: 4.0})
        assert TupleUnpacker.unpack(tup, ordered_required_items, ordered_optional_items) == tup + (None, )

        tup = ("id", 3, [3.0, 4.5], {"a": 1.0, "b": 2.0})
        expected = tup[:3] + (None, ) + tup[3:]
        assert TupleUnpacker.unpack(tup, ordered_required_items, ordered_optional_items) == expected

    def test_unpack_OnlyOptionalInInput_typeErrorThrown(self):
        tup = ({1: 2.0, 3: 4.0}, {"a": 1.0, "b": 2.0})
        ordered_required_items = [('id', str),
                                  ('int', int)]
        ordered_optional_items = [('sparse_vector', Dict[int, float]), ('metadata', Dict[str, Any])]

        with pytest.raises(TypeError,
                           match=re.escape(
                               "Argument 'id' in position 0 must be of type: <class 'str'>."
                               "\nReceived value: {1: 2.0, 3: 4.0}")):

            TupleUnpacker.unpack(tup, ordered_required_items, ordered_optional_items)

    def test_unpack_moreItemsThanExpected_typeErrorRaised(self):

        tup = ("id", 3, [3.0, 4.5], {1: 2.0, 3: 4.0}, {"a": 1.0, "b": 2.0}, "extra")
        ordered_required_items = [('id', str),
                                  ('int', int),
                                  ('vector', List[float])]
        ordered_optional_items = [('sparse_vector', Dict[int, float]), ('metadata', Dict[str, Any])]

        with pytest.raises(TypeError,
                           match=re.escape(
                               "Too many arguments in input tuple.\n "
                               "Expected at most: 5\n"
                               "Input tuple: ('id', 3, [3.0, 4.5], {1: 2.0, 3: 4.0}, {'a': 1.0, 'b': 2.0}, 'extra')")):

            TupleUnpacker.unpack(tup, ordered_required_items, ordered_optional_items)