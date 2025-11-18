from typing import TypedDict, Any
from pinecone.inference import RerankModel


class SearchRerankTypedDict(TypedDict):
    # """
    # SearchRerank represents a rerank request when searching within a specific namespace.
    # """

    model: str | RerankModel
    # model: str
    # """
    # The name of the [reranking model](https://docs.pinecone.io/guides/inference/understanding-inference#reranking-models) to use.
    # Required.
    # """

    rank_fields: list[str]
    # rank_fields: list[str]
    # """
    # The fields to use for reranking.
    # Required.
    # """

    top_n: int | None
    # """
    # The number of top results to return after reranking. Defaults to top_k.
    # Optional.
    # """

    parameters: dict[str, Any] | None
    # """
    # Additional model-specific parameters. Refer to the [model guide](https://docs.pinecone.io/guides/inference/understanding-inference#models)
    # for available model parameters.
    # Optional.
    # """

    query: str | None
    # """
    # The query to rerank documents against. If a specific rerank query is specified, it overwrites
    # the query input that was provided at the top level.
    # """
