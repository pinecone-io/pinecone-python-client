"""Dict-like access mixins for ``msgspec.Struct`` model classes.

Two mixins are provided:

- :class:`DictLikeStruct` — the minimal interface used by data-plane models
  (``__getitem__``, ``__setitem__``, ``get``, ``__contains__``, ``__iter__``,
  ``to_dict``). Avoids method names that collide with common data fields
  like ``values`` or ``keys``, so it is safe for models such as ``Vector`` or
  ``SparseValues`` that declare a field called ``values``.
- :class:`StructDictMixin` — extends :class:`DictLikeStruct` with the full
  ``keys``/``values``/``items``/``__len__`` surface that assistant-domain
  models need. Do not use this one on a struct that has a field named
  ``keys``, ``values``, or ``items``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

from msgspec import Struct


def _struct_to_dict_recursive(value: Any) -> Any:
    """Recursively convert Struct instances and nested containers to plain dicts."""
    if isinstance(value, Struct):
        return {
            field: _struct_to_dict_recursive(getattr(value, field))
            for field in value.__struct_fields__
        }
    if isinstance(value, list):
        return [_struct_to_dict_recursive(item) for item in value]
    if isinstance(value, dict):
        return {k: _struct_to_dict_recursive(v) for k, v in value.items()}
    return value


class DictLikeStruct(Struct):
    """Minimal dict-like access for ``msgspec.Struct`` subclasses.

    Provides ``__getitem__`` / ``__setitem__`` / ``get`` / ``__contains__``
    / ``__iter__`` (iterates field names) and ``to_dict``. Does *not*
    introduce ``keys`` / ``values`` / ``items`` methods, so it is safe for
    structs with fields named ``values`` or ``keys``.

    Defined as a :class:`msgspec.Struct` itself so concrete subclasses can
    opt into ``gc=False`` without inheriting a non-struct ``__dict__``.
    """

    __struct_fields__: ClassVar[tuple[str, ...]]

    def __getitem__(self, key: str) -> Any:
        """Return the field value for *key*; raise ``KeyError`` for unknown keys."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the field *key* to *value*; raise ``KeyError`` for unknown keys."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        setattr(self, key, value)

    def __contains__(self, key: object) -> bool:
        """Return ``True`` if *key* is a field name of this model."""
        return key in self.__struct_fields__

    def __iter__(self) -> Iterator[str]:
        """Iterate over field names, mirroring ``dict.__iter__``."""
        return iter(self.__struct_fields__)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key* if it exists, otherwise *default*."""
        if key in self.__struct_fields__:
            return getattr(self, key)
        return default

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert this model and all nested Struct fields to a plain dict."""
        return {
            field: _struct_to_dict_recursive(getattr(self, field))
            for field in self.__struct_fields__
        }


class StructDictMixin(DictLikeStruct):
    """Full dict-like access for ``msgspec.Struct`` subclasses.

    Adds ``__len__`` / ``keys`` / ``values`` / ``items`` on top of the
    minimal :class:`DictLikeStruct` surface. Do not use this mixin on a
    struct that declares a field called ``keys``, ``values``, or ``items``
    (use :class:`DictLikeStruct` instead).
    """

    def __len__(self) -> int:
        """Return the number of fields."""
        return len(self.__struct_fields__)

    def keys(self) -> tuple[str, ...]:
        """Return all field names."""
        return self.__struct_fields__

    def values(self) -> list[Any]:
        """Return all field values in declaration order."""
        return [getattr(self, field) for field in self.__struct_fields__]

    def items(self) -> list[tuple[str, Any]]:
        """Return ``(name, value)`` pairs for all fields in declaration order."""
        return [(field, getattr(self, field)) for field in self.__struct_fields__]


# Backwards-compatibility alias — the class was renamed from ``DictMixin`` to
# ``StructDictMixin``. Retained for legacy imports.
DictMixin = StructDictMixin
