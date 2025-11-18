from dataclasses import dataclass, field
from typing import cast

from .utils import DictLike
from pinecone.utils.response_info import ResponseInfo


@dataclass
class UpdateResponse(DictLike):
    """Response from an update operation.

    Attributes:
        matched_records: The number of records that matched the filter (if a filter was provided).
        _response_info: Response metadata including LSN headers.
    """

    matched_records: int | None = None
    _response_info: ResponseInfo = field(
        default_factory=lambda: cast(ResponseInfo, {"raw_headers": {}}), repr=True, compare=False
    )
