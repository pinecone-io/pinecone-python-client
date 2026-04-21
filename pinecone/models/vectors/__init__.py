"""Vector models subpackage with lazy loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pinecone.models.vectors.query_aggregator import (  # noqa: F401
        QueryNamespacesResults,
        QueryResultsAggregator,
    )
    from pinecone.models.vectors.responses import (  # noqa: F401
        DescribeIndexStatsResponse,
        FetchByMetadataResponse,
        FetchResponse,
        ListItem,
        ListResponse,
        NamespaceSummary,
        Pagination,
        QueryResponse,
        ResponseInfo,
        UpdateResponse,
        UpsertRecordsResponse,
        UpsertResponse,
    )
    from pinecone.models.vectors.search import (  # noqa: F401
        Hit,
        RerankConfig,
        SearchRecordsResponse,
        SearchResult,
        SearchUsage,
    )
    from pinecone.models.vectors.sparse import SparseValues  # noqa: F401
    from pinecone.models.vectors.usage import Usage  # noqa: F401
    from pinecone.models.vectors.vector import ScoredVector, Vector  # noqa: F401

_LAZY_IMPORTS: dict[str, str] = {
    "SparseValues": "pinecone.models.vectors.sparse",
    "Usage": "pinecone.models.vectors.usage",
    "Vector": "pinecone.models.vectors.vector",
    "ScoredVector": "pinecone.models.vectors.vector",
    "QueryNamespacesResults": "pinecone.models.vectors.query_aggregator",
    "QueryResultsAggregator": "pinecone.models.vectors.query_aggregator",
    "UpsertResponse": "pinecone.models.vectors.responses",
    "QueryResponse": "pinecone.models.vectors.responses",
    "FetchByMetadataResponse": "pinecone.models.vectors.responses",
    "FetchResponse": "pinecone.models.vectors.responses",
    "NamespaceSummary": "pinecone.models.vectors.responses",
    "DescribeIndexStatsResponse": "pinecone.models.vectors.responses",
    "ResponseInfo": "pinecone.models.response_info",
    "ListItem": "pinecone.models.vectors.responses",
    "ListResponse": "pinecone.models.vectors.responses",
    "Pagination": "pinecone.models.vectors.responses",
    "UpdateResponse": "pinecone.models.vectors.responses",
    "UpsertRecordsResponse": "pinecone.models.vectors.responses",
    "Hit": "pinecone.models.vectors.search",
    "RerankConfig": "pinecone.models.vectors.search",
    "SearchUsage": "pinecone.models.vectors.search",
    "SearchResult": "pinecone.models.vectors.search",
    "SearchRecordsResponse": "pinecone.models.vectors.search",
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
