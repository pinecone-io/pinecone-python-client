from dataclasses import dataclass, field
from typing import Dict, cast

from .vector import Vector
from .utils import DictLike
from pinecone.utils.response_info import ResponseInfo
from pinecone.core.openapi.db_data.models import Usage


@dataclass
class Pagination(DictLike):
    next: str


@dataclass
class FetchByMetadataResponse(DictLike):
    namespace: str
    vectors: Dict[str, Vector]
    usage: Usage | None = None
    pagination: Pagination | None = None
    _response_info: ResponseInfo = field(
        default_factory=lambda: cast(ResponseInfo, {"raw_headers": {}}), repr=True, compare=False
    )
