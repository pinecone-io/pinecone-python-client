from typing import NamedTuple, List


class Pagination(NamedTuple):
    next: str


class ListResponse(NamedTuple):
    namespace: str
    vectors: List
    pagination: Pagination | None
