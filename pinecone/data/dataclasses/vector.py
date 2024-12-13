from typing import Any, Dict, List, Optional, Union
from .sparse_values import SparseValues
from .utils import DictLike

from dataclasses import dataclass, field


@dataclass
class Vector(DictLike):
    id: str
    values: List[float] = field(default_factory=list)
    metadata: Optional[Dict[str, Union[str, List[str]]]] = None
    sparse_values: Optional[SparseValues] = None

    def __post_init__(self):
        if self.sparse_values is None and len(self.values) == 0:
            raise ValueError("The values and sparse_values fields cannot both be empty")

    def to_dict(self) -> Dict[str, Any]:
        vector_dict = {"id": self.id, "values": self.values}
        if self.metadata is not None:
            vector_dict["metadata"] = self.metadata
        if self.sparse_values is not None:
            vector_dict["sparse_values"] = self.sparse_values.to_dict()
        return vector_dict

    @staticmethod
    def from_dict(vector_dict: Dict[str, Any]) -> "Vector":
        return Vector(
            id=vector_dict["id"],
            values=vector_dict["values"],
            metadata=vector_dict.get("metadata"),
            sparse_values=SparseValues.from_dict(vector_dict.get("sparse_values")),
        )
