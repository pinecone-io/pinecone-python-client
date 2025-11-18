from dataclasses import dataclass, field
from typing import cast

from .utils import DictLike
from pinecone.utils.response_info import ResponseInfo
from pinecone.core.openapi.db_data.models import ScoredVector, Usage


@dataclass
class QueryResponse(DictLike):
    """Response from a query operation.

    Attributes:
        matches: List of matched vectors with scores.
        namespace: The namespace that was queried.
        usage: Usage information (optional).
        _response_info: Response metadata including LSN headers.
    """

    matches: list[ScoredVector]
    namespace: str
    usage: Usage | None = None
    _response_info: ResponseInfo = field(
        default_factory=lambda: cast(ResponseInfo, {"raw_headers": {}}), repr=True, compare=False
    )
