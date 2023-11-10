from typing import NamedTuple, Dict, Optional

class IndexStatus(NamedTuple):
    state: str
    ready: bool
    host: str

class IndexDescription(NamedTuple):
    name: str
    dimension: int
    metric: str
    replicas: int
    shards: int
    pods: int
    pod_type: str
    capacity_mode: str
    status: IndexStatus
    metadata_config: Optional[Dict]
