from __future__ import annotations

import logging
import warnings
from typing import Dict, Any, TYPE_CHECKING

from pinecone.openapi_support import ApiClient
from pinecone.core.openapi.inference.apis import InferenceApi
from .models import EmbeddingsList, RerankResult
from pinecone.core.openapi.inference import API_VERSION
from pinecone.utils import setup_openapi_client, PluginAware
from pinecone.utils import require_kwargs

from .inference_request_builder import (
    InferenceRequestBuilder,
    EmbedModel as EmbedModelEnum,
    RerankModel as RerankModelEnum,
)

if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration
    from .resources.sync.model import Model as ModelResource
    from .models import ModelInfo, ModelInfoList

logger = logging.getLogger(__name__)
""" :meta private: """


class Inference(PluginAware):
    """
    The ``Inference`` class configures and uses the Pinecone Inference API to generate embeddings and
    rank documents.

    It is generally not instantiated directly, but rather accessed through a parent ``Pinecone`` client
    object that is responsible for managing shared configurations.

    .. code-block:: python

        from pinecone import Pinecone

        pc = Pinecone()
        embeddings = pc.inference.embed(
            model="text-embedding-3-small",
            inputs=["Hello, world!"],
            parameters={"input_type": "passage", "truncate": "END"}
        )


    :param config: A ``pinecone.config.Config`` object, configured and built in the ``Pinecone`` class.
    :type config: ``pinecone.config.Config``, required
    """

    EmbedModel = EmbedModelEnum
    RerankModel = RerankModelEnum

    def __init__(
        self,
        config: "Config",
        openapi_config: "OpenApiConfiguration",
        pool_threads: int = 1,
        **kwargs,
    ) -> None:
        self._config = config
        """ :meta private: """

        self._openapi_config = openapi_config
        """ :meta private: """

        self._pool_threads = pool_threads
        """ :meta private: """

        self.__inference_api = setup_openapi_client(
            api_client_klass=ApiClient,
            api_klass=InferenceApi,
            config=config,
            openapi_config=openapi_config,
            pool_threads=self._pool_threads,
            api_version=API_VERSION,
        )

        self._model: "ModelResource" | None = None  # Lazy initialization
        """ :meta private: """

        super().__init__()  # Initialize PluginAware

    @property
    def config(self) -> "Config":
        """:meta private:"""
        # The config property is considered private, but the name cannot be changed to include underscore
        # without breaking compatibility with plugins in the wild.
        return self._config

    @property
    def openapi_config(self) -> "OpenApiConfiguration":
        """:meta private:"""
        warnings.warn(
            "The `openapi_config` property has been renamed to `_openapi_config`. It is considered private and should not be used directly. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._openapi_config

    @property
    def pool_threads(self) -> int:
        """:meta private:"""
        warnings.warn(
            "The `pool_threads` property has been renamed to `_pool_threads`. It is considered private and should not be used directly. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._pool_threads

    @property
    def model(self) -> "ModelResource":
        """
        Model is a resource that describes models available in the Pinecone Inference API.

        Currently you can get or list models.

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            # List all models
            models = pc.inference.model.list()

            # List models, with model type filtering
            models = pc.inference.model.list(type="embed")
            models = pc.inference.model.list(type="rerank")

            # List models, with vector type filtering
            models = pc.inference.model.list(vector_type="dense")
            models = pc.inference.model.list(vector_type="sparse")

            # List models, with both type and vector type filtering
            models = pc.inference.model.list(type="rerank", vector_type="dense")

            # Get details on a specific model
            model = pc.inference.model.get("text-embedding-3-small")

        """
        if self._model is None:
            from .resources.sync.model import Model as ModelResource

            self._model = ModelResource(
                inference_api=self.__inference_api,
                config=self._config,
                openapi_config=self._openapi_config,
                pool_threads=self._pool_threads,
            )
        return self._model

    def embed(
        self,
        model: EmbedModelEnum | str,
        inputs: str | list[Dict] | list[str],
        parameters: dict[str, Any] | None = None,
    ) -> EmbeddingsList:
        """
        Generates embeddings for the provided inputs using the specified model and (optional) parameters.

        :param model: The model to use for generating embeddings.
        :type model: str, required
        :param inputs: A list of items to generate embeddings for.
        :type inputs: list, required
        :param parameters: A dictionary of parameters to use when generating embeddings.
        :type parameters: dict, optional

        :return: ``EmbeddingsList`` object with keys ``data``, ``model``, and ``usage``. The ``data`` key contains a list of
          ``n`` embeddings, where ``n`` = len(inputs). Precision of returned embeddings is either
          float16 or float32, with float32 being the default. ``model`` key is the model used to generate the embeddings.
          ``usage`` key contains the total number of tokens used at request-time.
        :rtype: EmbeddingsList

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()
            inputs = ["Who created the first computer?"]
            outputs = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=inputs,
                parameters={"input_type": "passage", "truncate": "END"}
            )
            print(outputs)
            # EmbeddingsList(
            #     model='multilingual-e5-large',
            #     data=[
            #         {'values': [0.1, ...., 0.2]},
            #     ],
            #     usage={'total_tokens': 6}
            # )

        You can also use a single string input:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()
            output = pc.inference.embed(
                model="text-embedding-3-small",
                inputs="Hello, world!"
            )

        Or use the EmbedModel enum:

        .. code-block:: python

            from pinecone import Pinecone
            from pinecone.inference import EmbedModel

            pc = Pinecone()
            outputs = pc.inference.embed(
                model=EmbedModel.TEXT_EMBEDDING_3_SMALL,
                inputs=["Document 1", "Document 2"]
            )

        """
        request_body = InferenceRequestBuilder.embed_request(
            model=model, inputs=inputs, parameters=parameters
        )
        resp = self.__inference_api.embed(embed_request=request_body)
        return EmbeddingsList(resp)

    def rerank(
        self,
        model: RerankModelEnum | str,
        query: str,
        documents: list[str] | list[dict[str, Any]],
        rank_fields: list[str] = ["text"],
        return_documents: bool = True,
        top_n: int | None = None,
        parameters: dict[str, Any] | None = None,
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

        :return: ``RerankResult`` object with keys ``data`` and ``usage``. The ``data`` key contains a list of
          ``n`` documents, where ``n`` = ``top_n`` and type(n) = Document. The documents are sorted in order of
          relevance, with the first being the most relevant. The ``index`` field can be used to locate the document
          relative to the list of documents specified in the request. Each document contains a ``score`` key
          representing how close the document relates to the query.
        :rtype: RerankResult

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()
            result = pc.inference.rerank(
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
            print(result)
            # RerankResult(
            #     model='bge-reranker-v2-m3',
            #     data=[{
            #         index=3,
            #         score=0.020924192,
            #         document={
            #             text='Acme Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.'
            #         }
            #     },{
            #         index=1,
            #         score=0.00034464317,
            #         document={
            #             text='Software is still eating the world.'
            #         }
            #     }],
            #     usage={'rerank_units': 1}
            # )

        You can also use document dictionaries with custom fields:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()
            result = pc.inference.rerank(
                model="pinecone-rerank-v0",
                query="What is machine learning?",
                documents=[
                    {"text": "Machine learning is a subset of AI.", "category": "tech"},
                    {"text": "Cooking recipes for pasta.", "category": "food"},
                ],
                rank_fields=["text"],
                top_n=1
            )

        Or use the RerankModel enum:

        .. code-block:: python

            from pinecone import Pinecone
            from pinecone.inference import RerankModel

            pc = Pinecone()
            result = pc.inference.rerank(
                model=RerankModel.PINECONE_RERANK_V0,
                query="Your query here",
                documents=["doc1", "doc2", "doc3"]
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
        resp = self.__inference_api.rerank(rerank_request=rerank_request)
        return RerankResult(resp)

    @require_kwargs
    def list_models(
        self, *, type: str | None = None, vector_type: str | None = None
    ) -> "ModelInfoList":
        """
        List all available models.

        :param type: The type of model to list. Either "embed" or "rerank".
        :type type: str, optional

        :param vector_type: The type of vector to list. Either "dense" or "sparse".
        :type vector_type: str, optional

        :return: A list of models.
        :rtype: ModelInfoList

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            # List all models
            models = pc.inference.list_models()

            # List models, with model type filtering
            models = pc.inference.list_models(type="embed")
            models = pc.inference.list_models(type="rerank")

            # List models, with vector type filtering
            models = pc.inference.list_models(vector_type="dense")
            models = pc.inference.list_models(vector_type="sparse")

            # List models, with both type and vector type filtering
            models = pc.inference.list_models(type="rerank", vector_type="dense")

        """
        return self.model.list(type=type, vector_type=vector_type)

    @require_kwargs
    def get_model(self, model_name: str) -> "ModelInfo":
        """
        Get details on a specific model.

        :param model_name: The name of the model to get details on.
        :type model_name: str, required

        :return: A ModelInfo object.
        :rtype: ModelInfo

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()
            model_info = pc.inference.get_model(model_name="pinecone-rerank-v0")
            print(model_info)
            # {
            #     "model": "pinecone-rerank-v0",
            #     "short_description": "A state of the art reranking model that out-performs competitors on widely accepted benchmarks. It can handle chunks up to 512 tokens (1-2 paragraphs)",
            #     "type": "rerank",
            #     "supported_parameters": [
            #         {
            #             "parameter": "truncate",
            #             "type": "one_of",
            #             "value_type": "string",
            #             "required": false,
            #             "default": "END",
            #             "allowed_values": [
            #                 "END",
            #                 "NONE"
            #             ]
            #         }
            #     ],
            #     "modality": "text",
            #     "max_sequence_length": 512,
            #     "max_batch_size": 100,
            #     "provider_name": "Pinecone",
            #     "supported_metrics": []
            # }

        """
        return self.model.get(model_name=model_name)
