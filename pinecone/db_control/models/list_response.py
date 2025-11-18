from typing import NamedTuple, Optional, List

from pinecone.core.openapi.db_data.model.usage import Usage


class Pagination(NamedTuple):
    next: str


class ListResponse(NamedTuple):
    namespace: str
    vectors: List
    pagination: Optional[Pagination]
    usage: Usage
