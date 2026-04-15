"""Backwards-compatibility shim for :mod:`pinecone.db_data.dataclasses.utils`.

Re-exports classes that used to live at :mod:`pinecone.db_data.dataclasses.utils` before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

# XXX: missing canonical; shim carries the only definition.
# DictLike has no equivalent in the new SDK. This class is defined here so that
# legacy code inheriting from or checking against DictLike continues to work.
# A follow-up task should move this to the canonical side if a proper replacement
# is desired.

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any


class DictLike:
    """Base class providing dict-like access to dataclass fields.

    Legacy callers used this as a base class for objects that support
    ``__getitem__``, ``__setitem__``, ``__contains__``, ``get``, and iteration
    over field names.
    """

    def __getitem__(self, key: str) -> Any:
        """Return the field value for *key*."""
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}  # type: ignore[arg-type]
        if key not in fields:
            raise KeyError(key)
        return fields[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the field *key* to *value*."""
        if not hasattr(self, key):
            raise KeyError(key)
        setattr(self, key, value)

    def __contains__(self, key: object) -> bool:
        """Return True if *key* is a field name."""
        field_names = {f.name for f in dataclasses.fields(self)}  # type: ignore[arg-type]
        return key in field_names

    def __iter__(self) -> Iterator[str]:
        """Iterate over field names."""
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            yield f.name

    def get(self, key: str, default: Any = None) -> Any:
        """Return field value for *key*, or *default* if the key is not found."""
        try:
            return self[key]
        except KeyError:
            return default


__all__ = ["DictLike"]
