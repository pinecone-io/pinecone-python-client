from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict, TYPE_CHECKING, Literal
from typing_extensions import NotRequired, TypeAlias
from enum import Enum

from ..enums import CloudProvider, AwsRegion, GcpRegion, AzureRegion


class ScalingConfigManualDict(TypedDict, total=False):
    """TypedDict for manual scaling configuration."""

    shards: int
    replicas: int


class ReadCapacityDedicatedConfigDict(TypedDict):
    """TypedDict for dedicated read capacity configuration.

    Required fields: node_type, scaling
    Optional fields: manual
    """

    node_type: str  # Required: "t1" or "b1"
    scaling: str  # Required: "Manual" or other scaling types
    manual: NotRequired[ScalingConfigManualDict]  # Optional


class ReadCapacityOnDemandDict(TypedDict):
    """TypedDict for OnDemand read capacity mode."""

    mode: Literal["OnDemand"]


class ReadCapacityDedicatedDict(TypedDict):
    """TypedDict for Dedicated read capacity mode."""

    mode: Literal["Dedicated"]
    dedicated: ReadCapacityDedicatedConfigDict


ReadCapacityDict = ReadCapacityOnDemandDict | ReadCapacityDedicatedDict

if TYPE_CHECKING:
    from pinecone.core.openapi.db_control.model.read_capacity import ReadCapacity
    from pinecone.core.openapi.db_control.model.read_capacity_on_demand_spec import (
        ReadCapacityOnDemandSpec,
    )
    from pinecone.core.openapi.db_control.model.read_capacity_dedicated_spec import (
        ReadCapacityDedicatedSpec,
    )

    ReadCapacityType: TypeAlias = (
        ReadCapacityDict | ReadCapacity | ReadCapacityOnDemandSpec | ReadCapacityDedicatedSpec
    )
else:
    ReadCapacityType: TypeAlias = ReadCapacityDict | Any


class MetadataSchemaFieldConfig(TypedDict):
    """TypedDict for metadata schema field configuration."""

    filterable: bool


@dataclass(frozen=True)
class ServerlessSpec:
    cloud: str
    region: str
    read_capacity: ReadCapacityType | None = None
    schema: dict[str, MetadataSchemaFieldConfig] | None = None

    def __init__(
        self,
        cloud: CloudProvider | str,
        region: AwsRegion | GcpRegion | AzureRegion | str,
        read_capacity: ReadCapacityType | None = None,
        schema: dict[str, MetadataSchemaFieldConfig] | None = None,
    ):
        # Convert Enums to their string values if necessary
        object.__setattr__(self, "cloud", cloud.value if isinstance(cloud, Enum) else str(cloud))
        object.__setattr__(
            self, "region", region.value if isinstance(region, Enum) else str(region)
        )
        object.__setattr__(self, "read_capacity", read_capacity)
        object.__setattr__(self, "schema", schema)

    def asdict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"serverless": {"cloud": self.cloud, "region": self.region}}
        if self.read_capacity is not None:
            result["serverless"]["read_capacity"] = self.read_capacity
        if self.schema is not None:
            result["serverless"]["schema"] = {"fields": self.schema}
        return result
