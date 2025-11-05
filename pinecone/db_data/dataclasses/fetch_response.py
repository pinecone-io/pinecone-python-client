from dataclasses import dataclass, field
from typing import Dict, Optional

from .vector import Vector
from .utils import DictLike
from pinecone.utils.response_info import ResponseInfo


@dataclass
class FetchResponse(DictLike):
    namespace: str
    vectors: Dict[str, Vector]
    usage: Dict[str, int]
    _response_info: Optional[ResponseInfo] = field(default=None, repr=False, compare=False)
