"""TextQuery class for full-text search queries."""

from __future__ import annotations

from dataclasses import dataclass

from .utils import DictLike


@dataclass
class TextQuery(DictLike):
    """A text query for full-text search.

    Used as the ``score_by`` parameter in ``search_documents()`` to perform
    full-text search on a specified field.

    :param field: The name of the field to search.
    :param query: The search query string.
    :param boost: Optional boost factor for this query's score contribution.
    :param slop: Optional slop parameter for phrase queries, controlling
        how many positions apart terms can be.

    Example usage::

        from pinecone import TextQuery

        results = index.search_documents(
            namespace="movies",
            score_by=TextQuery(field="title", query='return "pink panther"'),
            top_k=10,
        )
    """

    field: str
    query: str
    boost: float | None = None
    slop: int | None = None

    def to_dict(self) -> dict:
        """Serialize to API format.

        :returns: Dictionary representation for the API.
        """
        result: dict = {"field": self.field, "query": self.query}
        if self.boost is not None:
            result["boost"] = self.boost
        if self.slop is not None:
            result["slop"] = self.slop
        return result
