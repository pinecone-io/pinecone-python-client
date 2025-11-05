from dataclasses import dataclass, field
from typing import Dict, cast

from .vector import Vector
from .utils import DictLike
from pinecone.utils.response_info import ResponseInfo


@dataclass
class FetchResponse(DictLike):
    namespace: str
    vectors: Dict[str, Vector]
    usage: Dict[str, int]
    _response_info: ResponseInfo = field(
        default_factory=lambda: cast(ResponseInfo, {"raw_headers": {}}), repr=True, compare=False
    )
