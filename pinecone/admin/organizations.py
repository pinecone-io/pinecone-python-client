"""Organizations namespace — placeholder for P-0058."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient


class Organizations:
    """Manage Pinecone organizations."""

    def __init__(self, *, http: HTTPClient) -> None:
        self._http = http
