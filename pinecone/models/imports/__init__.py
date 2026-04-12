"""Bulk import models subpackage with lazy loading."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pinecone.models.imports.list import ImportList  # noqa: F401
    from pinecone.models.imports.model import (  # noqa: F401
        ImportModel,
        StartImportResponse,
    )

_LAZY_IMPORTS: dict[str, str] = {
    "ImportModel": "pinecone.models.imports.model",
    "StartImportResponse": "pinecone.models.imports.model",
    "ImportList": "pinecone.models.imports.list",
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
