import pytest


@pytest.fixture
def vector_dim():
    return 8

@pytest.fixture
def vals1(vector_dim):
    return [0.1] * vector_dim


@pytest.fixture
def vals2(vector_dim):
    return [0.2] * vector_dim


@pytest.fixture
def sparse_indices_1():
    return [1, 8, 42]


@pytest.fixture
def sparse_values_1():
    return [0.8, 0.9, 0.42]


@pytest.fixture
def sparse_indices_2():
    return [1, 3, 5]


@pytest.fixture
def sparse_values_2():
    return [0.7, 0.3, 0.31415]


@pytest.fixture
def md1():
    return {"genre": "action", "year": 2021}


@pytest.fixture
def md2():
    return {"genre": "documentary", "year": 2020}


@pytest.fixture
def filter1():
    return {"genre": {"$in": ["action"]}}


@pytest.fixture
def filter2():
    return {"year": {"$eq": 2020}}
