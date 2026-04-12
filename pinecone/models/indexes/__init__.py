"""Index models subpackage with lazy loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pinecone.models.indexes.index import (  # noqa: F401
        ByocSpecInfo,
        IndexModel,
        IndexSpec,
        IndexStatus,
        PodSpecInfo,
        ServerlessSpecInfo,
    )
    from pinecone.models.indexes.list import IndexList  # noqa: F401
    from pinecone.models.indexes.specs import (  # noqa: F401
        ByocSpec,
        EmbedConfig,
        IntegratedSpec,
        PodSpec,
        ServerlessSpec,
    )

_LAZY_IMPORTS: dict[str, str] = {
    "ByocSpecInfo": "pinecone.models.indexes.index",
    "IndexModel": "pinecone.models.indexes.index",
    "IndexSpec": "pinecone.models.indexes.index",
    "IndexStatus": "pinecone.models.indexes.index",
    "PodSpecInfo": "pinecone.models.indexes.index",
    "ServerlessSpecInfo": "pinecone.models.indexes.index",
    "IndexList": "pinecone.models.indexes.list",
    "ServerlessSpec": "pinecone.models.indexes.specs",
    "PodSpec": "pinecone.models.indexes.specs",
    "ByocSpec": "pinecone.models.indexes.specs",
    "EmbedConfig": "pinecone.models.indexes.specs",
    "IntegratedSpec": "pinecone.models.indexes.specs",
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
