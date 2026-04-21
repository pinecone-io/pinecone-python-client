"""Pagination response models for assistant list operations."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, abbreviate_list, safe_display
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.model import AssistantModel


class ListAssistantsResponse(Struct, kw_only=True):
    """Paginated response for listing assistants.

    Attributes:
        assistants: The assistants returned in this page.
        next: Token for fetching the next page of results, or ``None``
            when no more pages exist.
    """

    assistants: list[AssistantModel]
    next: str | None = None

    @property
    def next_token(self) -> str | None:
        """Backwards-compatibility alias for :attr:`next`."""
        return self.next

    @safe_display
    def __repr__(self) -> str:  # type: ignore[override]
        return f"ListAssistantsResponse(count={len(self.assistants)}, next={self.next!r})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ListAssistantsResponse(...)")
            return
        preview = abbreviate_list(self.assistants, head=3, formatter=lambda a: a.name)
        with p.group(2, "ListAssistantsResponse(", ")"):
            p.breakable()
            p.text(f"count={len(self.assistants)},")
            p.breakable()
            p.text(f"next={self.next!r},")
            p.breakable()
            p.text(f"assistants={preview}")

    @safe_display
    def _repr_html_(self) -> str:
        next_display = self.next if self.next is not None else "—"
        builder = HtmlBuilder("ListAssistantsResponse")
        builder.row("Count:", len(self.assistants))
        builder.row("Next page token:", next_display)
        shown = self.assistants[:5]
        section_rows: list[tuple[str, Any]] = [(a.name, a.status) for a in shown]
        if len(self.assistants) > 5:
            section_rows.append(("...", f"{len(self.assistants) - 5} more"))
        builder.section("Assistants", section_rows)
        return builder.build()


class ListFilesResponse(Struct, kw_only=True):
    """Paginated response for listing assistant files.

    Attributes:
        files: The files returned in this page.
        next: Token for fetching the next page of results, or ``None``
            when no more pages exist.
    """

    files: list[AssistantFileModel]
    next: str | None = None

    @property
    def next_token(self) -> str | None:
        """Backwards-compatibility alias for :attr:`next`."""
        return self.next
