from typing import NamedTuple

class ListIndexMeta(NamedTuple):
    name: str
    dimension: int
    metric: str
    capacity_mode: str
    host: str
