from dataclasses import dataclass
from .utils import DictLike


@dataclass
class SearchQueryVector(DictLike):
    """
    SearchQueryVector represents the vector values used to query.
    """

    values: list[float] | None = None
    """
    The vector data included in the search request.
    Optional.
    """

    sparse_values: list[float] | None = None
    """
    The sparse embedding values to search with.
    Optional.
    """

    sparse_indices: list[int] | None = None
    """
    The sparse embedding indices to search with.
    Optional.
    """

    def as_dict(self) -> dict:
        """
        Returns the SearchQueryVector as a dictionary.
        """
        d = {
            "values": self.values,
            "sparse_values": self.sparse_values,
            "sparse_indices": self.sparse_indices,
        }
        return {k: v for k, v in d.items() if v is not None}
