from dataclasses import dataclass
from typing import Optional, Any, Dict


@dataclass
class SearchQuery:
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

    filter: Optional[Dict[str, Any]] = None
    """
    The filter to apply to the search.
    Optional.
    """

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns the SearchQuery as a dictionary.
        """
        d = {"inputs": self.inputs, "top_k": self.top_k, "filter": self.filter}
        return {k: v for k, v in d.items() if v is not None}
