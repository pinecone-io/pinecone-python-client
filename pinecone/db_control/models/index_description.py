from typing import NamedTuple, Literal


class PodSpecDefinition(NamedTuple):
    replicas: int
    shards: int
    pods: int
    pod_type: str
    environment: str
    metadata_config: dict | None


class ServerlessSpecDefinition(NamedTuple):
    cloud: str
    region: str


PodKey = Literal["pod"]
PodSpec = dict[PodKey, PodSpecDefinition]

ServerlessKey = Literal["serverless"]
ServerlessSpec = dict[ServerlessKey, ServerlessSpecDefinition]
