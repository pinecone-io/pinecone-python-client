from typing import Optional, Dict, List, Union, Any

from pinecone.core.openapi.inference.api.inference_api import AsyncioInferenceApi
from .models import EmbeddingsList, RerankResult

from .inference_request_builder import (
    InferenceRequestBuilder,
    EmbedModel as EmbedModelEnum,
    RerankModel as RerankModelEnum,
)


class AsyncioInference:
    """
    The `Inference` class configures and uses the Pinecone Inference API to generate embeddings and
    rank documents.

    :param config: A `pinecone.config.Config` object, configured and built in the Pinecone class.
    :type config: `pinecone.config.Config`, required
    """

    EmbedModel = EmbedModelEnum
    RerankModel = RerankModelEnum

    def __init__(self, api_client, **kwargs) -> None:
        self.api_client = api_client
        self.__inference_api = AsyncioInferenceApi(api_client)

    async def embed(
        self,
        model: str,
        inputs: Union[str, List[Dict], List[str]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingsList:
        """
        Generates embeddings for the provided inputs using the specified model and (optional) parameters.

        :param model: The model to use for generating embeddings.
        :type model: str, required

        :param inputs: A list of items to generate embeddings for.
        :type inputs: list, required

        :param parameters: A dictionary of parameters to use when generating embeddings.
        :type parameters: dict, optional

        :return: EmbeddingsList object with keys `data`, `model`, and `usage`. The `data` key contains a list of
        `n` embeddings, where `n` = len(inputs) and type(n) = Embedding. Precision of returned embeddings is either
        float16 or float32, with float32 being the default. `model` key is the model used to generate the embeddings.
        `usage` key contains the total number of tokens used at request-time.

        Example:
        >>> inputs = ["Who created the first computer?"]
        >>> outputs = await pc.inference.embed(model="multilingual-e5-large", inputs=inputs, parameters={"input_type": "passage", "truncate": "END"})
        >>> print(outputs)
        EmbeddingsList(
            model='multilingual-e5-large',
            data=[
                {'values': [0.1, ...., 0.2]},
              ],
            usage={'total_tokens': 6}
        )
        """
        request_body = InferenceRequestBuilder.embed_request(
            model=model, inputs=inputs, parameters=parameters
        )
        resp = await self.__inference_api.embed(embed_request=request_body)
        return EmbeddingsList(resp)

    async def rerank(
        self,
        model: str,
        query: str,
        documents: Union[List[str], List[Dict[str, Any]]],
        rank_fields: List[str] = ["text"],
        return_documents: bool = True,
        top_n: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> RerankResult:
        """
        Rerank documents with associated relevance scores that represent the relevance of each document
        to the provided query using the specified model.

        :param model: The model to use for reranking.
        :type model: str, required

        :param query: The query to compare with documents.
        :type query: str, required

        :param documents: A list of documents or strings to rank.
        :type documents: list, required

        :param rank_fields: A list of document fields to use for ranking. Defaults to ["text"].
        :type rank_fields: list, optional

        :param return_documents: Whether to include the documents in the response. Defaults to True.
        :type return_documents: bool, optional

        :param top_n: How many documents to return. Defaults to len(documents).
        :type top_n: int, optional

        :param parameters: A dictionary of parameters to use when ranking documents.
        :type parameters: dict, optional

        :return: RerankResult object with keys `data` and `usage`. The `data` key contains a list of
        `n` documents, where `n` = `top_n` and type(n) = Document. The documents are sorted in order of
        relevance, with the first being the most relevant. The `index` field can be used to locate the document
        relative to the list of documents specified in the request. Each document contains a `score` key
        representing how close the document relates to the query.

        Example:
        >>> result = await pc.inference.rerank(
                model="bge-reranker-v2-m3",
                query="Tell me about tech companies",
                documents=[
                    "Apple is a popular fruit known for its sweetness and crisp texture.",
                    "Software is still eating the world.",
                    "Many people enjoy eating apples as a healthy snack.",
                    "Acme Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
                    "An apple a day keeps the doctor away, as the saying goes.",
                ],
                top_n=2,
                return_documents=True,
            )
        >>> print(result)
        RerankResult(
          model='bge-reranker-v2-m3',
          data=[
            { index=3, score=0.020980744,
              document={text="Acme Inc. has rev..."} },
            { index=1, score=0.00034015716,
              document={text="Software is still..."} }
          ],
          usage={'rerank_units': 1}
        )
        """
        rerank_request = InferenceRequestBuilder.rerank(
            model=model,
            query=query,
            documents=documents,
            rank_fields=rank_fields,
            return_documents=return_documents,
            top_n=top_n,
            parameters=parameters,
        )
        resp = await self.__inference_api.rerank(rerank_request=rerank_request)
        return RerankResult(resp)
