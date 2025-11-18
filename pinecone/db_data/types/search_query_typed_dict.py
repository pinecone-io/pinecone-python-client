from typing import TypedDict, Any
from .search_query_vector_typed_dict import SearchQueryVectorTypedDict


class SearchQueryTypedDict(TypedDict):
    """
    SearchQuery represents the query when searching within a specific namespace.
    """

    inputs: dict[str, Any]
    """
    The input data to search with.
    Required.
    """

    top_k: int
    """
    The number of results to return with each search.
    Required.
    """

    filter: dict[str, Any] | None
    """
    The filter to apply to the search.
    Optional.
    """

    vector: SearchQueryVectorTypedDict | None
    """
    The vector values to search with. If provided, it overwrites the inputs.
    """

    id: str | None
    """
    The unique ID of the vector to be used as a query vector.
    """

    match_terms: dict[str, Any] | None
    """
    Specifies which terms must be present in the text of each search hit based on the specified strategy.
    The match is performed against the text field specified in the integrated index field_map configuration.
    Terms are normalized and tokenized into single tokens before matching, and order does not matter.
    Expected format: {"strategy": "all", "terms": ["term1", "term2", ...]}
    Currently only "all" strategy is supported, which means all specified terms must be present.

    **Limitations:** match_terms is only supported for sparse indexes with integrated embedding
    configured to use the pinecone-sparse-english-v0 model.
    Optional.
    """
