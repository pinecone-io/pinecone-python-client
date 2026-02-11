from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .serverless_spec import ReadCapacityType, MetadataSchemaFieldConfig


@dataclass(frozen=True)
class ByocSpec:
    """
    ByocSpec represents the configuration used to deploy a BYOC (Bring Your Own Cloud) index.

    To learn more about the options for each configuration, please see [Understanding Indexes](https://docs.pinecone.io/docs/indexes)
    """

    environment: str
    read_capacity: ReadCapacityType | None = None
    schema: dict[str, MetadataSchemaFieldConfig] | None = None

    def asdict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"byoc": {"environment": self.environment}}
        if self.read_capacity is not None:
            result["byoc"]["read_capacity"] = self.read_capacity
        if self.schema is not None:
            result["byoc"]["schema"] = {"fields": self.schema}
        return result
