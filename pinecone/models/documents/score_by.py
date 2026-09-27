"""The six ways a document search can score a document, and the union over them.

Every ``score_by`` clause is one of these six, told apart by its ``type`` tag:
``text`` for BM25 keyword scoring, ``query_string`` for a Lucene expression,
``dense_vector`` for similarity to an embedding, ``sparse_vector`` for similarity to
a sparse vector, and ``embed`` and ``sparse_embed`` for similarity to query text that
Pinecone embeds with the model of a ``string`` field declared with that aspect. Pass
them as typed instances or as plain dicts with a ``type`` key — either way each one
checks its required fields on construction, so a malformed clause is rejected at the
call site.

``text`` and ``query_string`` clauses can be combined in one search to score on several
signals at once. A vector clause, and an ``embed`` or ``sparse_embed`` clause, has to
stand alone.
"""

from __future__ import annotations

import warnings
from typing import Any

from msgspec import Struct

from pinecone.models.vectors.sparse import SparseValues

__all__ = [
    "DenseVectorQuery",
    "DocumentScoringMethod",
    "EmbedQuery",
    "QueryStringQuery",
    "SparseEmbedQuery",
    "SparseVectorQuery",
    "TextQuery",
]


class TextQuery(Struct, tag="text", tag_field="type", kw_only=True, omit_defaults=True):
    """Score documents by BM25 keyword relevance over named text fields.

    The clause to reach for when the words themselves matter — exact terms, names, codes
    — rather than meaning. Scoring is per field, so name every field you want searched.

    Attributes:
        query: The words to search for, e.g. ``"sparse index scoring"``.
        fields: The text fields to search across, e.g. ``["title", "chunk"]``. At least
            one is required.
        field: Deprecated single-field form of ``fields``. A value here is moved into
            ``fields`` with a :exc:`DeprecationWarning` and this attribute reads back as
            ``None``, so read ``fields`` whichever way you set it.

    Raises:
        ValueError: If ``fields`` ends up empty, or if both ``fields`` and ``field`` are
            given.

    Examples:
        >>> from pinecone.models.documents.score_by import TextQuery
        >>> TextQuery(query="sparse index scoring", fields=["title", "chunk"]).fields
        ['title', 'chunk']

    .. seealso::
       :class:`QueryStringQuery` — when you need boolean operators in the query itself.
    """

    query: str
    fields: list[str] | None = None
    field: str | None = None

    def __post_init__(self) -> None:
        if self.field is not None:
            if self.fields is not None:
                raise ValueError(
                    "TextQuery accepts `fields=[...]` or the deprecated `field=...`, but not both."
                )
            warnings.warn(
                "TextQuery `field` is deprecated; use `fields=[...]`.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.fields = [self.field]
            self.field = None
        if not self.fields:
            raise ValueError("Text scoring requires specifying at least one field")


class QueryStringQuery(
    Struct, tag="query_string", tag_field="type", kw_only=True, omit_defaults=True
):
    """Score documents by a Lucene query string, with ``AND``, ``OR`` and ``NOT``.

    Choose this over :class:`TextQuery` when the query itself needs structure — combining
    terms, excluding one, or scoping a clause to one field. Fields are named *inside* the
    query string as ``field_name:(clause)``; leave the qualifiers off and every
    text-searchable field is searched.

    Attributes:
        query: The Lucene expression, e.g. ``'title:(sparse OR hybrid) NOT draft'``.
        field: Not accepted here — use a qualifier in ``query`` instead.
        fields: Not accepted here — use a qualifier in ``query`` instead.

    Raises:
        ValueError: If ``field`` or ``fields`` is given. Passing either is rejected rather
            than ignored, because the field would silently not be applied.

    Examples:
        >>> from pinecone.models.documents.score_by import QueryStringQuery
        >>> QueryStringQuery(query="title:(sparse OR hybrid) NOT draft").query
        'title:(sparse OR hybrid) NOT draft'
    """

    query: str
    field: str | None = None
    fields: list[str] | None = None

    def __post_init__(self) -> None:
        if self.field is not None or self.fields is not None:
            raise ValueError(
                "'score_by' clauses of type 'query_string' must not specify 'field' or "
                "'fields'; use Lucene-style field qualifiers (e.g. 'field_name:value') "
                "in the query string instead"
            )


class DenseVectorQuery(Struct, tag="dense_vector", tag_field="type", kw_only=True):
    """Score documents by dense vector similarity to a query embedding.

    The clause for finding documents by meaning. It scores against one field, so the
    field named has to be declared ``dense_vector`` in the index schema, and ``values``
    has to be as long as that field's ``dimension``. A dense clause cannot be combined
    with any other scoring method in the same search.

    Attributes:
        field: The dense vector field to score against, e.g. ``"embedding"``.
        values: The query embedding, one float per dimension of that field.

    Raises:
        ValueError: If ``field`` is empty, or ``values`` is empty.

    Examples:
        The three floats here stand in for a full-length embedding.

        >>> from pinecone.models.documents.score_by import DenseVectorQuery
        >>> DenseVectorQuery(field="embedding", values=[0.12, 0.34, 0.56]).field
        'embedding'

    .. seealso::
       :class:`SparseVectorQuery` — the same idea against a sparse field.
    """

    field: str
    values: list[float]

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError(
                "dense_vector scoring requires specifying the name of exactly one field"
            )
        if not self.values:
            raise ValueError("dense_vector scoring requires a non-empty 'values' array")


class SparseVectorQuery(Struct, tag="sparse_vector", tag_field="type", kw_only=True):
    """Score documents by sparse vector similarity to a query sparse vector.

    The clause for finding documents by term overlap when you already hold a sparse
    encoding of the query. The field named has to be declared ``sparse_vector`` in the
    index schema, and, like a dense clause, this one cannot be combined with any other
    scoring method in the same search.

    Attributes:
        field: The sparse vector field to score against, e.g. ``"keywords"``.
        sparse_values: The query's sparse vector, as
            :class:`~pinecone.models.vectors.sparse.SparseValues`.

    Raises:
        ValueError: If ``field`` is empty.

    Examples:
        >>> from pinecone import SparseValues
        >>> from pinecone.models.documents.score_by import SparseVectorQuery
        >>> SparseVectorQuery(
        ...     field="keywords",
        ...     sparse_values=SparseValues(indices=[10, 42], values=[0.4, 0.9]),
        ... ).field
        'keywords'

    .. seealso::
       :class:`TextQuery` — when you have the words rather than a sparse encoding of them.
    """

    field: str
    sparse_values: SparseValues

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError(
                "sparse_vector scoring requires specifying the name of exactly one field"
            )


class EmbedQuery(Struct, tag="embed", tag_field="type", kw_only=True, omit_defaults=True):
    """Score documents by dense similarity to query text that Pinecone embeds.

    The clause for searching a ``string`` field declared with ``embed``: pass the query
    as text, and Pinecone embeds it with that field's model and read parameters, then
    scores it against the field's stored embeddings. You never handle a vector. Like a
    dense clause, it has to stand alone in ``score_by``.

    Attributes:
        query: The text to search for, e.g. ``"how do vector indexes scale"``.
        fields: The one ``string`` field that declares ``embed``, as a one-element list,
            e.g. ``["body"]``.
        field: Single-name form of ``fields``, e.g. ``"body"``. A value here is moved
            into ``fields`` and this attribute reads back as ``None``, so read ``fields``
            whichever way you set it.
        values: Not accepted here — Pinecone embeds ``query`` for you.
        sparse_values: Not accepted here — Pinecone embeds ``query`` for you.

    Raises:
        ValueError: If ``query`` is empty, if the clause does not name exactly one field,
            if both ``fields`` and ``field`` are given, or if ``values`` or
            ``sparse_values`` is given. A vector is rejected rather than ignored, because
            it would silently not be used.

    Examples:
        >>> from pinecone.models.documents.score_by import EmbedQuery
        >>> EmbedQuery(field="body", query="how do vector indexes scale").fields
        ['body']

    .. seealso::
       :class:`DenseVectorQuery` — to score a ``dense_vector`` field with a vector you
       embedded yourself.
    """

    query: str
    fields: list[str] | None = None
    field: str | None = None
    values: Any | None = None
    sparse_values: Any | None = None

    def __post_init__(self) -> None:
        _normalize_embed_clause(self, "embed", "dense_vector")


class SparseEmbedQuery(
    Struct, tag="sparse_embed", tag_field="type", kw_only=True, omit_defaults=True
):
    """Score documents by sparse similarity to query text that Pinecone embeds.

    The clause for searching a ``string`` field declared with ``sparse_embed``: pass the
    query as text, and Pinecone embeds it with that field's sparse model and read
    parameters. It has to stand alone in ``score_by``.

    Attributes:
        query: The text to search for, e.g. ``"vector database"``.
        fields: The one ``string`` field that declares ``sparse_embed``, as a
            one-element list, e.g. ``["body"]``.
        field: Single-name form of ``fields``, e.g. ``"body"``. A value here is moved
            into ``fields`` and this attribute reads back as ``None``, so read ``fields``
            whichever way you set it.
        values: Not accepted here — Pinecone embeds ``query`` for you.
        sparse_values: Not accepted here — Pinecone embeds ``query`` for you.

    Raises:
        ValueError: If ``query`` is empty, if the clause does not name exactly one field,
            if both ``fields`` and ``field`` are given, or if ``values`` or
            ``sparse_values`` is given. A vector is rejected rather than ignored, because
            it would silently not be used.

    Examples:
        >>> from pinecone.models.documents.score_by import SparseEmbedQuery
        >>> SparseEmbedQuery(fields=["body"], query="vector database").fields
        ['body']

    .. seealso::
       :class:`SparseVectorQuery` — to score a ``sparse_vector`` field with a sparse
       vector you encoded yourself.
    """

    query: str
    fields: list[str] | None = None
    field: str | None = None
    values: Any | None = None
    sparse_values: Any | None = None

    def __post_init__(self) -> None:
        _normalize_embed_clause(self, "sparse_embed", "sparse_vector")


def _normalize_embed_clause(
    clause: EmbedQuery | SparseEmbedQuery, kind: str, vector_kind: str
) -> None:
    if clause.field is not None:
        if clause.fields is not None:
            raise ValueError(
                f"{type(clause).__name__} accepts `fields=[...]` or `field=...`, but not both."
            )
        clause.fields = [clause.field]
        clause.field = None
    if not clause.fields or len(clause.fields) != 1 or not clause.fields[0]:
        raise ValueError(f"{kind} scoring requires specifying the name of exactly one field")
    if not clause.query:
        raise ValueError(f"{kind} scoring requires a non-empty 'query'")
    if clause.values is not None or clause.sparse_values is not None:
        raise ValueError(
            f"'{kind}' scoring embeds 'query' with the model of field '{clause.fields[0]}' "
            "and takes no vector. Remove 'values' and 'sparse_values' from the clause, or "
            f"use '{vector_kind}' scoring to score a vector you supply."
        )


DocumentScoringMethod = (
    TextQuery
    | QueryStringQuery
    | DenseVectorQuery
    | SparseVectorQuery
    | EmbedQuery
    | SparseEmbedQuery
)
