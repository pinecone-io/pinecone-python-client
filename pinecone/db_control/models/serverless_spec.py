from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
from enum import Enum

from ..enums import CloudProvider, AwsRegion, GcpRegion, AzureRegion


@dataclass(frozen=True)
class ServerlessSpec:
    cloud: str
    region: str
    read_capacity: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        cloud: Union[CloudProvider, str],
        region: Union[AwsRegion, GcpRegion, AzureRegion, str],
        read_capacity: Optional[Dict[str, Any]] = None,
    ):
        # Convert Enums to their string values if necessary
        object.__setattr__(self, "cloud", cloud.value if isinstance(cloud, Enum) else str(cloud))
        object.__setattr__(
            self, "region", region.value if isinstance(region, Enum) else str(region)
        )
        object.__setattr__(self, "read_capacity", read_capacity)

    def asdict(self):
        result = {"serverless": {"cloud": self.cloud, "region": self.region}}
        if self.read_capacity is not None:
            result["serverless"]["read_capacity"] = self.read_capacity
        return result
