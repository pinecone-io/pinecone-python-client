"""Factory functions for creating query objects.

These functions provide a simpler API for constructing query objects
to use with ``search_documents()``.
"""

from __future__ import annotations

from .dataclasses import TextQuery, VectorQuery, SparseValues


def text_query(
    field: str, query: str, boost: float | None = None, slop: int | None = None
) -> TextQuery:
    """Create a text query for full-text search.

    Factory function that creates a :class:`~pinecone.TextQuery` object
    for use with ``search_documents()``.

    :param field: The name of the field to search.
    :param query: The search query string. Supports:

        - Simple terms: ``"pink panther"``
        - Phrase matching: ``'"pink panther"'`` (quotes in string)
        - Required terms: ``"+return +panther"``

    :param boost: Optional boost factor for this query's score contribution.
    :param slop: Optional slop parameter for phrase queries, controlling
        how many positions apart terms can be.
    :returns: A TextQuery object.

    Example usage::

        from pinecone import text_query

        # Simple text search
        results = index.search_documents(
            namespace="movies",
            score_by=text_query("title", "pink panther"),
            top_k=10,
        )

        # Phrase match with quotes
        results = index.search_documents(
            namespace="movies",
            score_by=text_query("title", '"pink panther"'),
            top_k=10,
        )

        # With boost and slop
        results = index.search_documents(
            namespace="movies",
            score_by=text_query("title", '"pink panther"', boost=3.0, slop=2),
            top_k=10,
        )
    """
    return TextQuery(field=field, query=query, boost=boost, slop=slop)


def vector_query(
    field: str, values: list[float] | None = None, sparse_values: SparseValues | None = None
) -> VectorQuery:
    """Create a vector query for similarity search.

    Factory function that creates a :class:`~pinecone.VectorQuery` object
    for use with ``search_documents()``.

    :param field: The name of the vector field to search.
    :param values: Dense vector values for similarity search.
    :param sparse_values: Sparse vector values for hybrid search.
    :returns: A VectorQuery object.

    Example usage::

        from pinecone import vector_query, SparseValues

        # Dense vector search
        results = index.search_documents(
            namespace="movies",
            score_by=vector_query("embedding", values=[0.1, 0.2, 0.3]),
            top_k=10,
        )

        # Sparse vector search
        results = index.search_documents(
            namespace="movies",
            score_by=vector_query(
                "sparse_embedding",
                sparse_values=SparseValues(indices=[1, 5, 10], values=[0.5, 0.3, 0.2]),
            ),
            top_k=10,
        )
    """
    return VectorQuery(field=field, values=values, sparse_values=sparse_values)
