"""Dict-like access mixin for msgspec.Struct assistant models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from msgspec import Struct

if TYPE_CHECKING:
    pass


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


class StructDictMixin:
    """Mixin that adds dict-like read access to msgspec.Struct subclasses.

    Provides key subscript, membership testing, length, keys/values/items,
    safe access with default, recursive dict conversion, and string/repr as
    dictionary form.

    The concrete class must be a ``msgspec.Struct`` subclass so that
    ``__struct_fields__`` is available at runtime.
    """

    __struct_fields__: ClassVar[tuple[str, ...]]

    def __getitem__(self, key: str) -> Any:
        """Return the field value for *key*; raise ``KeyError`` for unknown keys."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Return ``True`` if *key* is a field name of this model."""
        return key in self.__struct_fields__

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

    def __str__(self) -> str:
        """Return string representation in dictionary form."""
        return str(self.to_dict())

    def __repr__(self) -> str:
        """Return repr in dictionary form."""
        return repr(self.to_dict())


# Backwards-compatibility alias — the class was renamed from DictMixin to StructDictMixin.
DictMixin = StructDictMixin
