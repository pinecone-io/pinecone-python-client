from dataclasses import dataclass, field
from typing import Optional

from .utils import DictLike
from pinecone.utils.response_info import ResponseInfo


@dataclass
class UpsertResponse(DictLike):
    """Response from an upsert operation.

    Attributes:
        upserted_count: Number of vectors that were upserted.
        _response_info: Response metadata including LSN headers (optional).
    """

    upserted_count: int
    _response_info: Optional[ResponseInfo] = field(default=None, repr=False, compare=False)
