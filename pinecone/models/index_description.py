from typing import NamedTuple, Dict, Optional, Union, Literal

class IndexStatus(NamedTuple):
    state: str
    ready: bool

PodKey = Literal['pod']
class PodSpecDefinition(NamedTuple):
    replicas: int
    shards: int
    pods: int
    pod_type: str
    environment: str
    metadata_config: Optional[Dict]

PodSpec = Dict[PodKey, PodSpecDefinition]

ServerlessKey = Literal['serverless']
class ServerlessSpecDefinition(NamedTuple):
    cloud: str
    region: str

ServerlessSpec = Dict[ServerlessKey, ServerlessSpecDefinition]

class IndexDescription(NamedTuple):
    name: str
    dimension: int
    metric: str
    host: str
    spec: Union[PodSpec, ServerlessSpec]
    status: IndexStatus
