from dataclasses import dataclass, field
from typing import cast

from .utils import DictLike
from pinecone.utils.response_info import ResponseInfo


@dataclass
class UpsertResponse(DictLike):
    """Response from an upsert operation.

    Attributes:
        upserted_count: Number of vectors that were upserted.
        _response_info: Response metadata including LSN headers.
    """

    upserted_count: int
    _response_info: ResponseInfo = field(
        default_factory=lambda: cast(ResponseInfo, {"raw_headers": {}}), repr=True, compare=False
    )
