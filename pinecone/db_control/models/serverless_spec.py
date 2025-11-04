from dataclasses import dataclass
from typing import Union, Optional, Dict, Any, TypedDict, TYPE_CHECKING, Literal
from enum import Enum

try:
    from typing_extensions import NotRequired
except ImportError:
    try:
        from typing import NotRequired  # type: ignore
    except ImportError:
        # Fallback for older Python versions - NotRequired not available
        NotRequired = None  # type: ignore

from ..enums import CloudProvider, AwsRegion, GcpRegion, AzureRegion

if TYPE_CHECKING:
    from pinecone.core.openapi.db_control.model.read_capacity import ReadCapacity
    from pinecone.core.openapi.db_control.model.read_capacity_on_demand_spec import (
        ReadCapacityOnDemandSpec,
    )
    from pinecone.core.openapi.db_control.model.read_capacity_dedicated_spec import (
        ReadCapacityDedicatedSpec,
    )


class ScalingConfigManualDict(TypedDict, total=False):
    """TypedDict for manual scaling configuration."""

    shards: int
    replicas: int


if NotRequired is not None:
    # Python 3.11+ or typing_extensions available - use NotRequired for better type hints
    class ReadCapacityDedicatedConfigDict(TypedDict):
        """TypedDict for dedicated read capacity configuration.

        Required fields: node_type, scaling
        Optional fields: manual
        """

        node_type: str  # Required: "t1" or "b1"
        scaling: str  # Required: "Manual" or other scaling types
        manual: NotRequired[ScalingConfigManualDict]  # Optional
else:
    # Fallback for older Python versions - all fields optional
    class ReadCapacityDedicatedConfigDict(TypedDict, total=False):  # type: ignore[no-redef]
        """TypedDict for dedicated read capacity configuration.

        Note: In older Python versions without NotRequired support, all fields
        are marked as optional. However, node_type and scaling are required
        when using Dedicated mode. Users must provide these fields.
        """

        node_type: str  # Required: "t1" or "b1"
        scaling: str  # Required: "Manual" or other scaling types
        manual: ScalingConfigManualDict  # Optional


class ReadCapacityOnDemandDict(TypedDict):
    """TypedDict for OnDemand read capacity mode."""

    mode: Literal["OnDemand"]


class ReadCapacityDedicatedDict(TypedDict):
    """TypedDict for Dedicated read capacity mode."""

    mode: Literal["Dedicated"]
    dedicated: ReadCapacityDedicatedConfigDict


ReadCapacityDict = Union[ReadCapacityOnDemandDict, ReadCapacityDedicatedDict]

if TYPE_CHECKING:
    ReadCapacityType = Union[
        ReadCapacityDict, "ReadCapacity", "ReadCapacityOnDemandSpec", "ReadCapacityDedicatedSpec"
    ]
else:
    ReadCapacityType = Union[ReadCapacityDict, Any]


class MetadataSchemaFieldConfig(TypedDict):
    """TypedDict for metadata schema field configuration."""

    filterable: bool


@dataclass(frozen=True)
class ServerlessSpec:
    cloud: str
    region: str
    read_capacity: Optional[ReadCapacityType] = None
    schema: Optional[Dict[str, MetadataSchemaFieldConfig]] = None

    def __init__(
        self,
        cloud: Union[CloudProvider, str],
        region: Union[AwsRegion, GcpRegion, AzureRegion, str],
        read_capacity: Optional[ReadCapacityType] = None,
        schema: Optional[Dict[str, MetadataSchemaFieldConfig]] = None,
    ):
        # Convert Enums to their string values if necessary
        object.__setattr__(self, "cloud", cloud.value if isinstance(cloud, Enum) else str(cloud))
        object.__setattr__(
            self, "region", region.value if isinstance(region, Enum) else str(region)
        )
        object.__setattr__(self, "read_capacity", read_capacity)
        object.__setattr__(self, "schema", schema)

    def asdict(self):
        result = {"serverless": {"cloud": self.cloud, "region": self.region}}
        if self.read_capacity is not None:
            result["serverless"]["read_capacity"] = self.read_capacity
        if self.schema is not None:
            result["serverless"]["schema"] = {"fields": self.schema}
        return result
