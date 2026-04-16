"""Preview document response models (2026-01.alpha API)."""

from __future__ import annotations

from typing import Any

import orjson
from msgspec import Struct

__all__ = ["PreviewDocument", "Usage"]


class Usage(Struct, kw_only=True):
    """API usage statistics for a preview search request.

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
