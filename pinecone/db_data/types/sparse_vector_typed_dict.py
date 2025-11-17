from typing import TypedDict


class SparseVectorTypedDict(TypedDict):
    indices: list[int]
    values: list[float]
