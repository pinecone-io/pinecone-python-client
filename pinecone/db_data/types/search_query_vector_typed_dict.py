from typing import TypedDict


class SearchQueryVectorTypedDict(TypedDict):
    """
    SearchQueryVector represents the vector values used to query.
    """

    values: list[float] | None
    """
    The vector data included in the search request.
    Optional.
    """

    sparse_values: list[float] | None
    """
    The sparse embedding values to search with.
    Optional.
    """

    sparse_indices: list[int] | None
    """
    The sparse embedding indices to search with.
    Optional.
    """
