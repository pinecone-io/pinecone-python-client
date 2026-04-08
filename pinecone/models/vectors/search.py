"""Search records response models."""

from __future__ import annotations

from typing import Any, Literal, TypedDict, overload

from msgspec import Struct

from pinecone.models.vectors.responses import ResponseInfo


class _RerankConfigRequired(TypedDict):
    """Required fields of :class:`RerankConfig`."""

    model: str
    rank_fields: list[str]


class RerankConfig(_RerankConfigRequired, total=False):
    """Typed configuration for the ``rerank`` parameter of :meth:`~pinecone.Index.search`.

    Required keys: ``model``, ``rank_fields``.
    All other keys are optional.

    Attributes:
        model (str): Reranking model name (e.g. ``"bge-reranker-v2-m3"``).
        rank_fields (list[str]): Record fields to rank on (e.g. ``["text"]``).
        top_n (int): Number of top results to return after reranking.
            Defaults to the value of ``top_k`` when omitted.
        parameters (dict[str, Any]): Model-specific parameters forwarded to
            the reranker. See the model documentation for supported keys.
        query (str): Override query text used for reranking.  When omitted the
            query is inferred from the search inputs.
    """

    top_n: int
    parameters: dict[str, Any]
    query: str


class SearchUsage(Struct, kw_only=True):
    """Usage statistics for a search operation.

    Attributes:
        read_units (int): Number of read units consumed.
        embed_total_tokens (int | None): Total tokens used for embedding, or ``None``
            if the search did not use integrated embedding.
        rerank_units (int | None): Number of rerank units consumed, or ``None`` if the
            search did not use reranking.
    """

    read_units: int
    embed_total_tokens: int | None = None
    rerank_units: int | None = None


class Hit(Struct, kw_only=True, rename={"id_": "_id", "score_": "_score"}):
    """A single search result hit.

    The API returns ``_id`` and ``_score`` as field names. These are mapped
    to ``id_`` and ``score_`` internally (to avoid Python name mangling),
    with convenience properties ``id`` and ``score`` for clean access.

    Attributes:
        id_ (str): The record identifier (wire name ``_id``).
        score_ (float): The similarity score (wire name ``_score``).
        fields (dict[str, Any]): Record fields included in the result.
    """

    id_: str
    score_: float
    fields: dict[str, Any] = {}

    @property
    def id(self) -> str:
        """Alias for ``id_`` to provide a cleaner API."""
        return self.id_

    @property
    def score(self) -> float:
        """Alias for ``score_`` to provide a cleaner API."""
        return self.score_

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. ``hit['id']``)."""
        if key == "id":
            return self.id_
        if key == "score":
            return self.score_
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'id' in hit``)."""
        if key in ("id", "score"):
            return True
        return key in self.__struct_fields__

    def __repr__(self) -> str:
        return f"Hit(id={self.id!r}, score={self.score!r}, fields={self.fields!r})"


class SearchResult(Struct, kw_only=True):
    """The result wrapper containing hits.

    Attributes:
        hits (list[Hit]): List of search result hits.
    """

    hits: list[Hit] = []


class SearchRecordsResponse(Struct, kw_only=True):
    """Response from a search records operation.

    Attributes:
        result (SearchResult): Wrapper containing the list of hits.
        usage (SearchUsage): Usage statistics for the search operation.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN values), or
            ``None`` if not populated.
    """

    result: SearchResult
    usage: SearchUsage
    response_info: ResponseInfo | None = None

    @overload
    def __getitem__(self, key: Literal["result"]) -> SearchResult: ...

    @overload
    def __getitem__(self, key: Literal["usage"]) -> SearchUsage: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. ``response['result']``)."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'result' in response``)."""
        return key in self.__struct_fields__
