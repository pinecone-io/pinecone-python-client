"""Collection models subpackage with lazy loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pinecone.models.collections.list import CollectionList  # noqa: F401
    from pinecone.models.collections.model import CollectionModel  # noqa: F401

_LAZY_IMPORTS: dict[str, str] = {
    "CollectionModel": "pinecone.models.collections.model",
    "CollectionList": "pinecone.models.collections.list",
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    """Lazy-load models on first access."""
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module = import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    import builtins

    return builtins.list({*globals(), *__all__, *_LAZY_IMPORTS})
