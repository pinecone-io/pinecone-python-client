"""Tests for inference response models."""

from __future__ import annotations

from pinecone.models.enums import RerankModel
from pinecone.models.inference import (
    DenseEmbedding,
    EmbeddingsList,
    EmbedUsage,
    ModelInfo,
    ModelInfoList,
    ModelInfoSupportedParameter,
    RankedDocument,
    RerankResult,
    RerankUsage,
    SparseEmbedding,
)


class TestDenseEmbedding:
    def test_dense_embedding_struct(self) -> None:
        emb = DenseEmbedding(values=[0.1, 0.2, 0.3])
        assert emb.values == [0.1, 0.2, 0.3]
        assert emb.vector_type == "dense"
        assert emb["values"] == [0.1, 0.2, 0.3]
        assert emb["vector_type"] == "dense"

    def test_dense_embedding_bracket_access_missing_key(self) -> None:
        emb = DenseEmbedding(values=[1.0])
        try:
            emb["nonexistent"]
            assert False, "Expected KeyError"
        except KeyError:
            pass


class TestSparseEmbedding:
    def test_sparse_embedding_struct(self) -> None:
        emb = SparseEmbedding(
            sparse_values=[0.5, 0.8],
            sparse_indices=[10, 42],
        )
        assert emb.sparse_values == [0.5, 0.8]
        assert emb.sparse_indices == [10, 42]
        assert emb.sparse_tokens is None
        assert emb.vector_type == "sparse"

    def test_sparse_embedding_with_tokens(self) -> None:
        emb = SparseEmbedding(
            sparse_values=[0.5, 0.8],
            sparse_indices=[10, 42],
            sparse_tokens=["hello", "world"],
        )
        assert emb.sparse_tokens == ["hello", "world"]
        assert emb["sparse_tokens"] == ["hello", "world"]


class TestEmbeddingsList:
    def test_embeddings_list_indexing(self) -> None:
        embeddings = [
            DenseEmbedding(values=[0.1, 0.2]),
            DenseEmbedding(values=[0.3, 0.4]),
            DenseEmbedding(values=[0.5, 0.6]),
        ]
        usage = EmbedUsage(total_tokens=15)
        result = EmbeddingsList(
            model="multilingual-e5-large",
            vector_type="dense",
            data=embeddings,
            usage=usage,
        )

        assert len(result) == 3
        assert result[0].values == [0.1, 0.2]
        assert result[2].values == [0.5, 0.6]

        items = list(result)
        assert len(items) == 3
        assert items[1].values == [0.3, 0.4]

    def test_embeddings_list_bracket_access(self) -> None:
        usage = EmbedUsage(total_tokens=5)
        result = EmbeddingsList(
            model="multilingual-e5-large",
            vector_type="dense",
            data=[DenseEmbedding(values=[1.0])],
            usage=usage,
        )
        assert result["model"] == "multilingual-e5-large"
        assert result["vector_type"] == "dense"
        assert result["usage"].total_tokens == 5

    def test_embeddings_list_with_sparse(self) -> None:
        embeddings = [
            SparseEmbedding(sparse_values=[0.5], sparse_indices=[10]),
        ]
        usage = EmbedUsage(total_tokens=3)
        result = EmbeddingsList(
            model="pinecone-sparse-english-v0",
            vector_type="sparse",
            data=embeddings,
            usage=usage,
        )
        assert len(result) == 1
        assert result[0].sparse_values == [0.5]


class TestRerankResult:
    def test_rerank_result_struct(self) -> None:
        docs = [
            RankedDocument(index=2, score=0.95, document={"text": "best match"}),
            RankedDocument(index=0, score=0.72, document={"text": "okay match"}),
            RankedDocument(index=1, score=0.31),
        ]
        usage = RerankUsage(rerank_units=1)
        result = RerankResult(model="bge-reranker-v2-m3", data=docs, usage=usage)

        assert result.data[0].score == 0.95
        assert result.data[0].document == {"text": "best match"}
        assert result.data[2].document is None
        assert result.usage.rerank_units == 1
        assert result["model"] == "bge-reranker-v2-m3"

    def test_ranked_document_bracket_access(self) -> None:
        doc = RankedDocument(index=0, score=0.9)
        assert doc["index"] == 0
        assert doc["score"] == 0.9
        assert doc["document"] is None


class TestModelInfo:
    def test_model_info_struct(self) -> None:
        param = ModelInfoSupportedParameter(
            parameter="input_type",
            type="one_of",
            value_type="string",
            required=False,
            allowed_values=["passage", "query"],
        )
        model = ModelInfo(
            model="multilingual-e5-large",
            short_description="A multilingual embedding model",
            type="embed",
            supported_parameters=[param],
            vector_type="dense",
            default_dimension=1024,
            supported_dimensions=[384, 512, 768, 1024],
            modality="text",
            max_sequence_length=512,
            max_batch_size=96,
            provider_name="pinecone",
            supported_metrics=["cosine", "dotproduct", "euclidean"],
        )

        assert model.model == "multilingual-e5-large"
        assert model.type == "embed"
        assert model.default_dimension == 1024
        assert model.supported_dimensions == [384, 512, 768, 1024]
        assert len(model.supported_parameters) == 1
        assert model.supported_parameters[0].parameter == "input_type"
        assert model["model"] == "multilingual-e5-large"
        assert model["vector_type"] == "dense"

    def test_model_info_optional_fields(self) -> None:
        model = ModelInfo(
            model="bge-reranker-v2-m3",
            short_description="A reranking model",
            type="rerank",
            supported_parameters=[],
        )
        assert model.vector_type is None
        assert model.default_dimension is None
        assert model.supported_dimensions is None
        assert model.modality is None
        assert model.max_sequence_length is None
        assert model.max_batch_size is None
        assert model.provider_name is None
        assert model.supported_metrics is None

    def test_model_info_supported_parameter_all_fields(self) -> None:
        param = ModelInfoSupportedParameter(
            parameter="truncate",
            type="one_of",
            value_type="string",
            required=False,
            allowed_values=["END", "NONE"],
            min=0.0,
            max=1.0,
            default="END",
        )
        assert param["parameter"] == "truncate"
        assert param["allowed_values"] == ["END", "NONE"]
        assert param["min"] == 0.0
        assert param["max"] == 1.0
        assert param["default"] == "END"


class TestModelInfoList:
    def test_model_info_list_names(self) -> None:
        models = [
            ModelInfo(
                model="multilingual-e5-large",
                short_description="Embedding model",
                type="embed",
                supported_parameters=[],
            ),
            ModelInfo(
                model="bge-reranker-v2-m3",
                short_description="Reranking model",
                type="rerank",
                supported_parameters=[],
            ),
        ]
        model_list = ModelInfoList(models)

        assert model_list.names() == ["multilingual-e5-large", "bge-reranker-v2-m3"]
        assert model_list[0].model == "multilingual-e5-large"
        assert model_list[1].model == "bge-reranker-v2-m3"
        assert model_list["models"] == models
        assert model_list.models == models
        assert len(model_list) == 2

        items = list(model_list)
        assert len(items) == 2

    def test_model_info_list_empty(self) -> None:
        model_list = ModelInfoList([])
        assert model_list.names() == []
        assert len(model_list) == 0
        assert list(model_list) == []

    def test_model_info_list_repr(self) -> None:
        model_list = ModelInfoList([])
        assert "ModelInfoList" in repr(model_list)

    def test_model_info_list_invalid_key(self) -> None:
        model_list = ModelInfoList([])
        try:
            model_list["invalid"]
            assert False, "Expected KeyError"
        except KeyError:
            pass


class TestRerankModelEnum:
    def test_rerank_model_enum(self) -> None:
        assert RerankModel.BGE_RERANKER_V2_M3.value == "bge-reranker-v2-m3"
        assert RerankModel.COHERE_RERANK_3_5.value == "cohere-rerank-3.5"
        assert RerankModel.PINECONE_RERANK_V0.value == "pinecone-rerank-v0"

    def test_rerank_model_is_str(self) -> None:
        assert isinstance(RerankModel.BGE_RERANKER_V2_M3, str)
        assert RerankModel.BGE_RERANKER_V2_M3 == "bge-reranker-v2-m3"
