"""Backwards-compatibility shim for :mod:`pinecone.models.assistant.list`.

Re-exports :class:`ListAssistantsResponse` that used to live at
:mod:`pinecone_plugins.assistant.models.list_assistants_response` before the
``python-sdk2`` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from dataclasses import dataclass
from typing import Any, Optional

from pinecone_plugins.assistant.models.core.dataclass import BaseDataclass


@dataclass
class ListAssistantsResponse(BaseDataclass):
    """Paginated list of assistants."""

    assistants: list[Any]
    next_token: Optional[str]

    @classmethod
    def from_openapi(cls, resp: Any, client_builder: Any, config: Any) -> "ListAssistantsResponse":
        from pinecone_plugins.assistant.models.assistant_model import (
            AssistantModel,  # type: ignore[import-untyped]
        )

        assistants = [
            AssistantModel(assistant=a, client_builder=client_builder, config=config)
            for a in (resp.assistants or [])
        ]
        next_token: Optional[str] = None
        pagination = getattr(resp, "pagination", None)
        if pagination is not None:
            next_token = getattr(pagination, "next", None)
        return cls(assistants=assistants, next_token=next_token)


__all__ = ["ListAssistantsResponse"]
