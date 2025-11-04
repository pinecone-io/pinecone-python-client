from dataclasses import dataclass
from typing import Dict, Optional

from .vector import Vector


@dataclass
class Pagination:
    next: str


@dataclass
class FetchByMetadataResponse:
    namespace: str
    vectors: Dict[str, Vector]
    usage: Dict[str, int]
    pagination: Optional[Pagination] = None
