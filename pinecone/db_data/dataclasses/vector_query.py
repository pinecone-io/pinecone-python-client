"""VectorQuery class for vector similarity search queries."""

from __future__ import annotations

from dataclasses import dataclass

from .utils import DictLike
from .sparse_values import SparseValues


@dataclass
class VectorQuery(DictLike):
    """A vector query for similarity search.

    Used as the ``score_by`` parameter in ``search_documents()`` to perform
    vector similarity search on a specified field.

    :param field: The name of the vector field to search.
    :param values: Dense vector values for similarity search.
    :param sparse_values: Sparse vector values for hybrid search.

    Example usage::

        from pinecone import VectorQuery

        # Dense vector query
        results = index.search_documents(
            namespace="movies",
            score_by=VectorQuery(field="embedding", values=[0.1, 0.2, 0.3, ...]),
            top_k=10,
        )

        # Sparse vector query
        results = index.search_documents(
            namespace="movies",
            score_by=VectorQuery(
                field="sparse_embedding",
                sparse_values=SparseValues(indices=[1, 5, 10], values=[0.5, 0.3, 0.2]),
            ),
            top_k=10,
        )
    """

    field: str
    values: list[float] | None = None
    sparse_values: SparseValues | None = None

    def to_dict(self) -> dict:
        """Serialize to API format.

        :returns: Dictionary representation for the API.
        """
        result: dict = {"field": self.field}
        if self.values is not None:
            result["values"] = self.values
        if self.sparse_values is not None:
            result["sparse_values"] = self.sparse_values.to_dict()
        return result
