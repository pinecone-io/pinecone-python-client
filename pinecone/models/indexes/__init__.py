"""Index models subpackage with lazy loading."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pinecone.models.indexes.index import IndexModel, IndexStatus  # noqa: F401
    from pinecone.models.indexes.list import IndexList  # noqa: F401
    from pinecone.models.indexes.specs import ByocSpec, PodSpec, ServerlessSpec  # noqa: F401

_LAZY_IMPORTS: dict[str, str] = {
    "IndexModel": "pinecone.models.indexes.index",
    "IndexStatus": "pinecone.models.indexes.index",
    "IndexList": "pinecone.models.indexes.list",
    "ServerlessSpec": "pinecone.models.indexes.specs",
    "PodSpec": "pinecone.models.indexes.specs",
    "ByocSpec": "pinecone.models.indexes.specs",
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    """Lazy-load models on first access."""
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module = import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
