from dataclasses import dataclass

from typing import Any, Dict, List
from .utils import DictLike


@dataclass
class SparseValues(DictLike):
    indices: List[int]
    values: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {"indices": self.indices, "values": self.values}

    @staticmethod
    def from_dict(sparse_values_dict: Dict[str, Any]) -> "SparseValues":
        return SparseValues(
            indices=sparse_values_dict["indices"], values=sparse_values_dict["values"]
        )
