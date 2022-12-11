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
        expected = {key: v for (key, t), v in zip(ordered_required_items, tup)}
        assert TupleUnpacker(ordered_required_items, []).unpack(tup) == expected

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

            TupleUnpacker(ordered_required_items, []).unpack(tup)

    def test_unpack_OnlyRequieredFieldsWrongType_ThrowException(self):
        ordered_required_items = [('id', str),
                                  ('int', int),
                                  ('vector', List[float]),
                                  ('sparse_vector', Dict[int, float]),
                                  ('metadata', Dict[str, Any])]

        tup = (7, 3, [3.0, 4.5], {1: 2.0, 3: 4.0}, {"a": 1.0, "b": 2.0})
        with pytest.raises(TypeError,
                           match=r"Argument 'id' in position 0 must be of type: <class 'str'>.\nReceived value: 7"):

            TupleUnpacker(ordered_required_items, []).unpack(tup)

        tup = ("id", 3, [3, 4], {1: 2.0, 3: 4.0}, {"a": 1.0, "b": 2.0})
        with pytest.raises(TypeError,
                           match=re.escape("Argument 'vector' in position 2 must be of type: typing.List[float]."
                                           "\nReceived value: [3, 4]")):

            TupleUnpacker(ordered_required_items, []).unpack(tup)

        tup = ("id", 3, [3.0, 4.5], {"1": 2.0, 3: 4.0}, {"a": 1, "b": 2.0})
        with pytest.raises(
                TypeError,
                match=re.escape("Argument 'sparse_vector' in position 3 must be of type: typing.Dict[int, float]."
                                "\nReceived value: {'1': 2.0, 3: 4.0}")):

            TupleUnpacker(ordered_required_items, []).unpack(tup)

    def test_unpack_RequiredAndOptionalAllOptionalInInput_ReturnInputTuple(self):
        tup = ("id", 3, [3.0, 4.5], {1: 2.0, 3: 4.0}, {"a": 1.0, "b": 2.0})
        ordered_required_items = [('id', str),
                                  ('int', int),
                                  ('vector', List[float])]
        ordered_optional_items = [('sparse_vector', Dict[int, float], {}), ('metadata', Dict[str, Any], {})]

        expected = {key: v for (key, t), v in zip(ordered_required_items, tup)}
        expected.update({key: v for (key, t, m), v in zip(ordered_optional_items, tup[3:])})
        assert TupleUnpacker(ordered_required_items, ordered_optional_items).unpack(tup) == expected

    def test_unpack_RequiredAndOptionalNoOptionalInInput_ReturnInputTupleWithMissingValues(self):
        tup = ("id", 3, [3.0, 4.5])
        ordered_required_items = [('id', str),
                                  ('int', int),
                                  ('vector', List[float])]
        ordered_optional_items = [('sparse_vector', Dict[int, float], {}), ('metadata', Dict[str, Any], {})]

        expected = {key: v for (key, t), v in zip(ordered_required_items, tup)}
        expected['sparse_vector'] = {}
        expected['metadata'] = {}
        assert TupleUnpacker(ordered_required_items, ordered_optional_items).unpack(tup) == expected

    def test_unpackRequiredAndOptionalAllOptionalInInputWrongType_ThrowException(self):
        tup = ("id", 3, [3.0, 4.5], {1: 2.0, 3: 4.0}, {1: 1, "b": 2.0})
        ordered_required_items = [('id', str),
                                  ('int', int),
                                  ('vector', List[float])]
        ordered_optional_items = [('sparse_vector', Dict[int, float], {}), ('metadata', Dict[str, Any], {})]

        with pytest.raises(
                TypeError,
                match=re.escape("Unexpected argument in position 4.\n"
                                "Input tuple: ('id', 3, [3.0, 4.5], {1: 2.0, 3: 4.0}, {1: 1, 'b': 2.0})")):

            TupleUnpacker(ordered_required_items, ordered_optional_items).unpack(tup)

    def test_unpack_onlySomeOfOptionalInInput_FillMissingValuesWithExpectedKeys(self):
        ordered_required_items = [('id', str),
                                  ('int', int),
                                  ('vector', List[float])]
        ordered_optional_items = [('sparse_vector', Dict[int, float], {}), ('metadata', Dict[str, Any], {})]

        tup = ("id", 3, [3.0, 4.5], {1: 2.0, 3: 4.0})
        expected = {key: v for (key, t), v in zip(ordered_required_items, tup)}
        expected['sparse_vector'] = tup[-1]
        expected['metadata'] = {}
        assert TupleUnpacker(ordered_required_items, ordered_optional_items).unpack(tup) == expected

        tup = ("id", 3, [3.0, 4.5], {"a": 1.0, "b": 2.0})
        expected['metadata'] = tup[-1]
        expected['sparse_vector'] = {}
        assert TupleUnpacker(ordered_required_items, ordered_optional_items).unpack(tup) == expected

    def test_unpack_OnlyOptionalInInput_typeErrorThrown(self):
        tup = ({1: 2.0, 3: 4.0}, {"a": 1.0, "b": 2.0})
        ordered_required_items = [('id', str),
                                  ('int', int)]
        ordered_optional_items = [('sparse_vector', Dict[int, float], {}), ('metadata', Dict[str, Any], {})]

        with pytest.raises(TypeError,
                           match=re.escape(
                               "Argument 'id' in position 0 must be of type: <class 'str'>."
                               "\nReceived value: {1: 2.0, 3: 4.0}")):

            TupleUnpacker(ordered_required_items, ordered_optional_items).unpack(tup)

    def test_unpack_moreItemsThanExpected_typeErrorRaised(self):

        tup = ("id", 3, [3.0, 4.5], {1: 2.0, 3: 4.0}, {"a": 1.0, "b": 2.0}, "extra")
        ordered_required_items = [('id', str),
                                  ('int', int),
                                  ('vector', List[float])]
        ordered_optional_items = [('sparse_vector', Dict[int, float], {}), ('metadata', Dict[str, Any], {})]

        with pytest.raises(TypeError,
                           match=re.escape(
                               "Too many arguments in input tuple.\n "
                               "Expected at most: 5\n"
                               "Input tuple: ('id', 3, [3.0, 4.5], {1: 2.0, 3: 4.0}, {'a': 1.0, 'b': 2.0}, 'extra')")):

            TupleUnpacker(ordered_required_items, ordered_optional_items).unpack(tup)
