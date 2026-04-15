"""Backwards-compatibility shim for the legacy DictMixin abstract base class.

Re-exports the abstract dict-access protocol that used to live at
:mod:`pinecone_plugins.assistant.models.core.dict_mixin` before the
``python-sdk2`` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from abc import ABC, abstractmethod
from collections.abc import ItemsView, KeysView, ValuesView
from typing import Any


class DictMixin(ABC):
    """Abstract mixin providing dict-like read access to model objects.

    Concrete subclasses must implement :meth:`to_dict`.
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` representation of this object."""
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        data = self.to_dict()
        if key not in data:
            raise KeyError(f"Key '{key}' not found in the object.")
        return data[key]

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()

    def __len__(self) -> int:
        return len(self.to_dict())

    def keys(self) -> KeysView[str]:
        return self.to_dict().keys()

    def values(self) -> ValuesView[Any]:
        return self.to_dict().values()

    def items(self) -> ItemsView[str, Any]:
        return self.to_dict().items()

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __str__(self) -> str:
        return str(self.to_dict())

    def __repr__(self) -> str:
        return str(self.to_dict())


__all__ = ["DictMixin"]
