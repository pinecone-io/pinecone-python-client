from typing import TypedDict, List


class SearchQueryVectorTypedDict(TypedDict):
    """
    SearchQueryVector represents the vector values used to query.
    """

    values: List[float] | None
    """
    The vector data included in the search request.
    Optional.
    """

    sparse_values: List[float] | None
    """
    The sparse embedding values to search with.
    Optional.
    """

    sparse_indices: List[int] | None
    """
    The sparse embedding indices to search with.
    Optional.
    """
