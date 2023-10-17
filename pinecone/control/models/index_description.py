from typing import NamedTuple


class IndexDescription(NamedTuple):
    name: str
    metric: str
    replicas: int
    dimension: int
    shards: int
    pods: int
    pod_type: str
    status: None
    metadata_config: None

