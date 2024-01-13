from typing import NamedTuple, Optional, Dict

class PodSpec(NamedTuple):
    environment: str
    replicas: Optional[int] = None
    shards: Optional[int] = None
    pods: Optional[int] = None
    pod_type: Optional[str] = "p1.x1"
    metadata_config: Optional[Dict] = {}

    def asdict(self):
        return {"pod": self._asdict()}