"""Preview document response models (2026-01.alpha API)."""

from __future__ import annotations

from typing import Any

import orjson
from msgspec import Struct

from pinecone.models._display import render_table
from pinecone.models.response_info import ResponseInfo

__all__ = [
    "PreviewDocument",
    "PreviewDocumentFetchResponse",
    "PreviewDocumentSearchResponse",
    "PreviewDocumentUpsertResponse",
    "PreviewUsage",
]


class PreviewUsage(Struct, kw_only=True):
    """API usage statistics for a preview search request.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        read_units: Number of read units consumed by the request.
    """

    read_units: int


class PreviewDocument:
    """A document returned from a preview search operation.

    Provides typed access to ``_id`` and ``_score`` fields, plus attribute-style
    and dict-style access to arbitrary document fields from ``include_fields``.

    The ``id``, ``_id``, and ``score`` typed properties always take precedence
    over document fields with the same names — a document field named ``"_score"``
    is only reachable via ``.get("_score")`` or ``.to_dict()["_score"]``.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        id: The document's unique identifier.
        _id: Alias for ``id``.
        score: Relevance score, or ``None`` when not present in the response.
        _score: Alias for ``score``.

    Examples:
        >>> from pinecone import Pinecone
        >>> pc = Pinecone(api_key="your-api-key")
        >>> index = pc.preview.index(host="https://articles-en-preview.svc.pinecone.io")
        >>> results = index.documents.search(
        ...     namespace="articles-en",
        ...     top_k=5,
        ...     score_by=[{"field": "chunk_text", "query": "climate change"}],
        ... )
        >>> doc = results.matches[0]
        >>> doc.id
        'article-42'
        >>> doc.score
        0.891

        Read custom document fields via attribute access:

        >>> doc.title  # doctest: +SKIP
        'Ocean acidification and climate change'
        >>> doc.category  # doctest: +SKIP
        'science'

        Read a field via dict-style access:

        >>> doc.get("title")  # doctest: +SKIP
        'Ocean acidification and climate change'
        >>> doc.get("missing_field", "n/a")
        'n/a'
    """

    __slots__ = ("_data",)
    _data: dict[str, Any]

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", data)

    @property
    def id(self) -> str:
        value = self._data.get("_id")
        if not value:
            raise AttributeError("Document has no '_id' field")
        return str(value)

    @property
    def _id(self) -> str:
        return self.id

    @property
    def score(self) -> float | None:
        raw = self._data.get("_score")
        if raw is None:
            return None
        return float(raw)

    @property
    def _score(self) -> float | None:
        return self.score

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key* from the document, or *default*.

        Equivalent to :meth:`dict.get` on the underlying document data.
        Typed properties (``_id``, ``_score``) are reachable via ``.get()``
        alongside any custom fields returned by the operation.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Args:
            key (str): Name of the document field to retrieve.
            default (Any): Value to return when *key* is absent
                (default: ``None``).

        Returns:
            The field value, or *default* if the field is not present.

        Examples:
            >>> doc = results.matches[0]
            >>> doc.get("category")
            'science'
            >>> doc.get("missing_field", "n/a")
            'n/a'
        """
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return the document as a plain dictionary.

        Returns a shallow copy of the underlying document data, including
        ``_id``, ``_score``, and all custom fields from the operation.
        Mutating the returned dict does not affect the document.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Returns:
            :class:`dict` mapping field names to their values.

        Examples:
            >>> doc = results.matches[0]
            >>> data = doc.to_dict()
            >>> data["_id"]
            'article-42'
            >>> data["title"]
            'Ocean acidification and climate change'
        """
        return dict(self._data)

    def to_json(self) -> str:
        """Return the document as a JSON string.

        Serializes the document using orjson. The result is a decoded UTF-8
        string suitable for writing to a file or logging.

        .. admonition:: Preview
           :class: warning

           Uses Pinecone API version ``2026-01.alpha``.
           Preview surface is not covered by SemVer — signatures and behavior
           may change in any minor SDK release. Pin your SDK version when
           relying on preview features.

        Returns:
            :class:`str` — a compact JSON-encoded string (decoded UTF-8)
            containing all document fields, including ``_id``, ``_score``,
            and any custom fields from the operation.

        Examples:
            >>> doc = results.matches[0]
            >>> json_str = doc.to_json()
            >>> import json; json.loads(json_str)["_id"]
            'article-42'
        """
        return orjson.dumps(self._data).decode()

    def __getattr__(self, name: str) -> Any:
        # __slots__ attributes are resolved before __getattr__, so _data is safe.
        # Properties (id, _id, score) are descriptors on the class and normally
        # take precedence over __getattr__. But when a property raises
        # AttributeError, Python falls back to __getattr__; block those reserved
        # names here so a user-defined `id`/`_id`/`score` field cannot leak
        # through the typed property's failure path.
        if name in ("id", "_id", "score"):
            raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __repr__(self) -> str:
        _id = self._data.get("_id", "")
        score = self._data.get("_score")
        extras = {k: v for k, v in self._data.items() if k not in ("_id", "_score")}
        if extras:
            return f"PreviewDocument(_id={_id!r}, score={score!r}, ...)"
        return f"PreviewDocument(_id={_id!r}, score={score!r})"


class PreviewDocumentUpsertResponse(Struct, kw_only=True):
    """Response from a document upsert operation.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        upserted_count: Number of documents successfully upserted.
        response_info: HTTP response metadata (request ID and LSN headers),
            or ``None`` when not present.
    """

    upserted_count: int
    response_info: ResponseInfo | None = None


class PreviewDocumentSearchResponse:
    """Response from a document search operation.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        matches: Ordered list of matching documents.
        namespace: The namespace that was searched.
        usage: API usage statistics, or ``None`` when not returned.
        response_info: HTTP response metadata (request ID and LSN headers),
            or ``None`` when not present.
    """

    __slots__ = ("matches", "namespace", "response_info", "usage")
    matches: list[PreviewDocument]
    namespace: str
    usage: PreviewUsage | None
    response_info: ResponseInfo | None

    def __init__(
        self,
        matches: list[PreviewDocument],
        namespace: str,
        usage: PreviewUsage | None = None,
        response_info: ResponseInfo | None = None,
    ) -> None:
        object.__setattr__(self, "matches", matches)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "usage", usage)
        object.__setattr__(self, "response_info", response_info)

    def __repr__(self) -> str:
        return (
            f"SearchResponse(matches={len(self.matches)}, "
            f"namespace={self.namespace!r}, "
            f"usage={self.usage!r})"
        )

    def _repr_html_(self) -> str:
        rows: list[tuple[str, str | int | float]] = [
            ("Matches:", len(self.matches)),
            ("Namespace:", self.namespace),
        ]
        if self.usage is not None:
            rows.append(("Read Units:", self.usage.read_units))
        return render_table("SearchResponse", rows)


class PreviewDocumentFetchResponse:
    """Response from a document fetch operation.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        documents: Map of document ID to document.
        namespace: The namespace that was fetched from.
        usage: API usage statistics, or ``None`` when not returned.
        response_info: HTTP response metadata (request ID and LSN headers),
            or ``None`` when not present.
    """

    __slots__ = ("documents", "namespace", "response_info", "usage")
    documents: dict[str, PreviewDocument]
    namespace: str
    usage: PreviewUsage | None
    response_info: ResponseInfo | None

    def __init__(
        self,
        documents: dict[str, PreviewDocument],
        namespace: str,
        usage: PreviewUsage | None = None,
        response_info: ResponseInfo | None = None,
    ) -> None:
        object.__setattr__(self, "documents", documents)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "usage", usage)
        object.__setattr__(self, "response_info", response_info)

    def __repr__(self) -> str:
        return (
            f"FetchResponse(documents={len(self.documents)}, namespace={self.namespace!r},"
            f" usage={self.usage!r})"
        )
