from typing import NamedTuple, Dict, Literal


class PodSpecDefinition(NamedTuple):
    replicas: int
    shards: int
    pods: int
    pod_type: str
    environment: str
    metadata_config: Dict | None


class ServerlessSpecDefinition(NamedTuple):
    cloud: str
    region: str


PodKey = Literal["pod"]
PodSpec = dict[PodKey, PodSpecDefinition]

ServerlessKey = Literal["serverless"]
ServerlessSpec = dict[ServerlessKey, ServerlessSpecDefinition]
