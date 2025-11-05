from dataclasses import dataclass, field
from typing import List, Optional

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
        _response_info: Response metadata including LSN headers (optional).
    """

    matches: List[ScoredVector]
    namespace: str
    usage: Optional[Usage] = None
    _response_info: Optional[ResponseInfo] = field(default=None, repr=True, compare=False)
