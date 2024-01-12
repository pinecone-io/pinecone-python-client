import pytest
from pinecone.utils import convert_to_list
from pinecone import SparseValues
import numpy as np
import pandas as pd

def test_convert_to_list_when_numpy_array():
    obj = np.array([1, 2, 3])
    actual = convert_to_list(obj)
    expected = [1, 2, 3]
    assert actual == expected
    assert actual[0].__class__ == expected[0].__class__

def test_convert_to_list_when_pandas_array():
    obj = pd.array([1, 2, 3])
    actual = convert_to_list(obj)
    expected = [1, 2, 3]
    assert actual == expected
    assert actual[0].__class__ == expected[0].__class__

def test_convert_to_list_when_pandas_float_array():
    obj = pd.array([0.1, 0.2, 0.3])
    actual = convert_to_list(obj)
    expected = [0.1, 0.2, 0.3]
    assert actual == expected
    assert actual[0].__class__ == expected[0].__class__

def test_convert_to_list_when_pandas_series():
    obj = pd.Series([1, 2, 3])
    actual = convert_to_list(obj)
    expected = [1, 2, 3]
    assert actual == expected
    assert actual[0].__class__ == expected[0].__class__

def test_convert_to_list_when_already_list():
    obj = [1, 2, 3]
    actual = convert_to_list(obj)
    expected = [1, 2, 3]
    assert actual == expected
