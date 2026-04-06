"""ApiKeys namespace — placeholder for follow-up task."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient


class ApiKeys:
    """Manage Pinecone API keys."""

    def __init__(self, *, http: HTTPClient) -> None:
        self._http = http
