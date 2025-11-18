from .sparse_vector_typed_dict import SparseVectorTypedDict
from typing import TypedDict


class VectorTypedDict(TypedDict, total=False):
    values: list[float]
    metadata: dict
    sparse_values: SparseVectorTypedDict
    id: str
