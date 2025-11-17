from enum import Enum
from typing import Any

from pinecone.core.openapi.inference.models import (
    EmbedRequest,
    EmbedRequestInputs,
    Document,
    RerankRequest,
)
from pinecone.utils import convert_enum_to_string


class EmbedModel(Enum):
    Multilingual_E5_Large = "multilingual-e5-large"
    Pinecone_Sparse_English_V0 = "pinecone-sparse-english-v0"


class RerankModel(Enum):
    Bge_Reranker_V2_M3 = "bge-reranker-v2-m3"
    Cohere_Rerank_3_5 = "cohere-rerank-3.5"
    Pinecone_Rerank_V0 = "pinecone-rerank-v0"


class InferenceRequestBuilder:
    @staticmethod
    def embed_request(
        model: EmbedModel | str,
        inputs: str | list[dict] | list[str],
        parameters: dict[str, Any] | None = None,
    ) -> EmbedRequest:
        model = convert_enum_to_string(model)
        embeddings_inputs: list[EmbedRequestInputs] = []
        if isinstance(inputs, str):
            embeddings_inputs = [EmbedRequestInputs(text=inputs)]
        elif isinstance(inputs, list) and len(inputs) > 0:
            if isinstance(inputs[0], str):
                embeddings_inputs = [EmbedRequestInputs(text=i) for i in inputs]
            elif isinstance(inputs[0], dict):
                embeddings_inputs = [EmbedRequestInputs(**i) for i in inputs]
            else:
                raise Exception("Invalid type for variable 'inputs'")
        else:
            raise Exception("Invalid type for variable 'inputs'")

        from typing import cast

        if parameters:
            result = EmbedRequest(model=model, inputs=embeddings_inputs, parameters=parameters)
            return cast(EmbedRequest, result)
        else:
            result = EmbedRequest(model=model, inputs=embeddings_inputs)
            return cast(EmbedRequest, result)

    @staticmethod
    def rerank(
        model: RerankModel | str,
        query: str,
        documents: list[str] | list[dict[str, Any]],
        rank_fields: list[str] = ["text"],
        return_documents: bool = True,
        top_n: int | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> RerankRequest:
        if isinstance(model, RerankModel):
            model = model.value
        else:
            model = str(model)

        if isinstance(documents, list) and len(documents) > 0:
            if isinstance(documents[0], str):
                documents = [Document(text=doc) for doc in documents]
            elif isinstance(documents[0], dict):
                documents = [Document(**doc) for doc in documents]
            else:
                raise Exception("Invalid type for variable 'documents'")
        else:
            raise Exception("Invalid type or value for variable 'documents'")

        args: dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": documents,
            "rank_fields": rank_fields,
            "return_documents": return_documents,
        }
        if top_n is not None:
            args["top_n"] = top_n
        if parameters is not None:
            args["parameters"] = parameters

        from typing import cast

        result = RerankRequest(**args)
        return cast(RerankRequest, result)
