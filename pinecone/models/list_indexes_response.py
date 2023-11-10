from typing import NamedTuple, List
from .list_index_meta import ListIndexMeta

class ListIndexesResponse(NamedTuple):
    databases: List[ListIndexMeta]