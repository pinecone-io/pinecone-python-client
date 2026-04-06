"""Projects namespace — placeholder for P-0059."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient


class Projects:
    """Manage Pinecone projects."""

    def __init__(self, *, http: HTTPClient) -> None:
        self._http = http
