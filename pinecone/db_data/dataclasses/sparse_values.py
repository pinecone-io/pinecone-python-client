from dataclasses import dataclass

from .utils import DictLike
from ..types import SparseVectorTypedDict


@dataclass
class SparseValues(DictLike):
    indices: list[int]
    values: list[float]

    def to_dict(self) -> SparseVectorTypedDict:
        return {"indices": self.indices, "values": self.values}

    @staticmethod
    def from_dict(sparse_values_dict: SparseVectorTypedDict) -> "SparseValues":
        return SparseValues(
            indices=sparse_values_dict["indices"], values=sparse_values_dict["values"]
        )
