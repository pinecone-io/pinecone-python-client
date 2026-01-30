"""DocumentSearchResponse class for document search responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from .document import Document
from .utils import DictLike
from pinecone.utils.response_info import ResponseInfo
from pinecone.core.openapi.db_data.models import Usage


@dataclass
class DocumentSearchResponse(DictLike):
    """Response from a document search operation.

    :param documents: List of documents matching the search query.
    :param usage: Usage information for the request.

    Example usage::

        results = index.search_documents(
            namespace="movies",
            score_by=text_query("title", "pink panther"),
            top_k=10,
        )

        print(f"Found {len(results.documents)} documents")
        print(f"Read units: {results.usage.read_units}")

        for doc in results.documents:
            print(f"{doc.id}: {doc.score}")
    """

    documents: list[Document]
    usage: Usage | None = None
    _response_info: ResponseInfo = field(
        default_factory=lambda: cast(ResponseInfo, {"raw_headers": {}}), repr=True, compare=False
    )
