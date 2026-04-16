"""Score-by query models for preview document search."""

from __future__ import annotations

from msgspec import Struct

from pinecone.models.vectors.sparse import SparseValues

__all__ = [
    "DenseVectorQuery",
    "QueryStringQuery",
    "ScoreByQuery",
    "SparseVectorQuery",
    "TextQuery",
]

_ADMONITION = """
    .. admonition:: Preview
       :class: warning

       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.
"""


class TextQuery(Struct, tag="text", tag_field="type", kw_only=True):
    """Full-text search query for scoring documents.

    .. admonition:: Preview
       :class: warning

       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        field: Name of the text field to search.
        query: Search query string.
    """

    field: str
    query: str


class QueryStringQuery(Struct, tag="query_string", tag_field="type", kw_only=True):
    """Query string syntax search with boolean operators (AND, OR, NOT).

    .. admonition:: Preview
       :class: warning

       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        query: Query string with operators (field embedded in query string per spec).
    """

    query: str


class DenseVectorQuery(Struct, tag="dense_vector", tag_field="type", kw_only=True):
    """Dense vector similarity query for scoring documents.

    .. admonition:: Preview
       :class: warning

       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        field: Name of the field containing dense vectors to search.
        values: Query vector as a list of floats.
    """

    field: str
    values: list[float]


class SparseVectorQuery(Struct, tag="sparse_vector", tag_field="type", kw_only=True):
    """Sparse vector similarity query for scoring documents.

    .. admonition:: Preview
       :class: warning

       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        field: Name of the field containing sparse vectors to search.
        sparse_values: Sparse vector with indices and values.
    """

    field: str
    sparse_values: SparseValues


ScoreByQuery = TextQuery | QueryStringQuery | DenseVectorQuery | SparseVectorQuery
