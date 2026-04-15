"""Backwards-compatibility shim for the legacy BaseDataclass.

Re-exports the base dataclass that used to live at
:mod:`pinecone_plugins.assistant.models.core.dataclass` before the
``python-sdk2`` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

from pinecone_plugins.assistant.models.core.dict_mixin import DictMixin


@dataclass
class BaseDataclass(DictMixin):
    """Base dataclass providing :class:`DictMixin` dict-access to all model fields."""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


__all__ = ["BaseDataclass"]
