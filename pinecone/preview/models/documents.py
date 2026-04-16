"""Preview document response models (2026-01.alpha API)."""

from __future__ import annotations

from typing import Any

import orjson
from msgspec import Struct

from pinecone.models._display import render_table

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

    Provides typed access to ``_id`` and ``score`` fields, plus attribute-style
    and dict-style access to arbitrary document fields from ``include_fields``.

    The ``id``, ``_id``, and ``score`` typed properties always take precedence
    over document fields with the same names — a document field named ``"score"``
    is only reachable via ``.get("score")`` or ``.to_dict()["score"]``.

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
    """

    __slots__ = ("_data",)
    _data: dict[str, Any]

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", data)

    @property
    def id(self) -> str:
        value = self._data.get("_id") or self._data.get("id")
        if not value:
            raise AttributeError("Document has no '_id' or 'id' field")
        return str(value)

    @property
    def _id(self) -> str:
        return self.id

    @property
    def score(self) -> float | None:
        raw = self._data.get("score")
        if raw is None:
            return None
        return float(raw)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def to_json(self) -> str:
        return orjson.dumps(self._data).decode()

    def __getattr__(self, name: str) -> Any:
        # __slots__ attributes are resolved before __getattr__, so _data is safe.
        # Properties (id, _id, score) are descriptors on the class and also
        # take precedence over __getattr__.
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __repr__(self) -> str:
        _id = self._data.get("_id") or self._data.get("id", "")
        score = self._data.get("score")
        extras = {k: v for k, v in self._data.items() if k not in ("_id", "id", "score")}
        if extras:
            return f"PreviewDocument(_id={_id!r}, score={score!r}, ...)"
        return f"PreviewDocument(_id={_id!r}, score={score!r})"


class PreviewDocumentUpsertResponse(Struct, kw_only=True):
    """Response from a document upsert operation.

    Attributes:
        upserted_count: Number of documents successfully upserted.
    """

    upserted_count: int


class PreviewDocumentSearchResponse:
    """Response from a document search operation.

    Attributes:
        matches: Ordered list of matching documents.
        namespace: The namespace that was searched.
        usage: API usage statistics, or ``None`` when not returned.
    """

    __slots__ = ("matches", "namespace", "usage")

    def __init__(
        self,
        matches: list[PreviewDocument],
        namespace: str,
        usage: PreviewUsage | None = None,
    ) -> None:
        object.__setattr__(self, "matches", matches)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "usage", usage)

    def __repr__(self) -> str:
        return (
            f"SearchResponse(matches={len(self.matches)}, "  # type: ignore[attr-defined]
            f"namespace={self.namespace!r}, "  # type: ignore[attr-defined]
            f"usage={self.usage!r})"  # type: ignore[attr-defined]
        )

    def _repr_html_(self) -> str:
        matches: list[PreviewDocument] = object.__getattribute__(self, "matches")
        namespace: str = object.__getattribute__(self, "namespace")
        usage: PreviewUsage | None = object.__getattribute__(self, "usage")
        rows: list[tuple[str, str | int | float]] = [
            ("Matches:", len(matches)),
            ("Namespace:", namespace),
        ]
        if usage is not None:
            rows.append(("Read Units:", usage.read_units))
        return render_table("SearchResponse", rows)


class PreviewDocumentFetchResponse:
    """Response from a document fetch operation.

    Attributes:
        documents: Map of document ID to document.
        namespace: The namespace that was fetched from.
        usage: API usage statistics, or ``None`` when not returned.
    """

    __slots__ = ("documents", "namespace", "usage")

    def __init__(
        self,
        documents: dict[str, PreviewDocument],
        namespace: str,
        usage: PreviewUsage | None = None,
    ) -> None:
        object.__setattr__(self, "documents", documents)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "usage", usage)

    def __repr__(self) -> str:
        documents: dict[str, PreviewDocument] = object.__getattribute__(self, "documents")
        namespace: str = object.__getattribute__(self, "namespace")
        usage: PreviewUsage | None = object.__getattribute__(self, "usage")
        return (
            f"FetchResponse(documents={len(documents)}, "
            f"namespace={namespace!r}, "
            f"usage={usage!r})"
        )
