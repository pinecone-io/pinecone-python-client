from typing import TypedDict, Optional, Dict, Any


class SearchQueryTypedDict(TypedDict):
    """
    SearchQuery represents the query when searching within a specific namespace.
    """

    inputs: Dict[str, Any]
    """
    The input data to search with.
    Required.
    """

    top_k: int
    """
    The number of results to return with each search.
    Required.
    """

    filter: Optional[Dict[str, Any]]
    """
    The filter to apply to the search.
    Optional.
    """
