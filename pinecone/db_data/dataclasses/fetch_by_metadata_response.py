from dataclasses import dataclass, field
from typing import Dict, Optional

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
    _response_info: Optional[ResponseInfo] = field(default=None, repr=False, compare=False)
