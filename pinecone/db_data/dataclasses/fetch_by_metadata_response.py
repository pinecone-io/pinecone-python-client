from dataclasses import dataclass, field
from typing import Dict, Optional, cast

from .vector import Vector
from .utils import DictLike
from pinecone.utils.response_info import ResponseInfo


@dataclass
class Pagination(DictLike):
    next: str


@dataclass
class FetchByMetadataResponse(DictLike):
    namespace: str
    vectors: Dict[str, Vector]
    usage: Dict[str, int]
    pagination: Optional[Pagination] = None
    _response_info: ResponseInfo = field(
        default_factory=lambda: cast(ResponseInfo, {"raw_headers": {}}), repr=True, compare=False
    )
