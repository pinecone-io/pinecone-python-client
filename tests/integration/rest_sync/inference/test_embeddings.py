import pytest
from pinecone import Pinecone, PineconeApiException, EmbedModel


class TestEmbed:
    @pytest.mark.parametrize(
        "model_input,model_output",
        [
            (EmbedModel.Multilingual_E5_Large, "multilingual-e5-large"),
            ("multilingual-e5-large", "multilingual-e5-large"),
        ],
    )
    def test_create_embeddings(self, model_input, model_output):
        pc = Pinecone()
        embeddings = pc.inference.embed(
            model=model_input,
            inputs=["The quick brown fox jumps over the lazy dog.", "lorem ipsum"],
            parameters={"input_type": "query", "truncate": "END"},
        )
        assert embeddings.vector_type == "dense"
        assert embeddings.get("vector_type") == "dense"
        assert embeddings.model == model_output
        assert embeddings.get("model") == model_output
        assert len(embeddings.data) == 2
        assert len(embeddings.get("data")) == 2
        assert embeddings.usage is not None

        individual_embedding = embeddings[0]
        assert len(individual_embedding.values) == 1024
        assert individual_embedding.vector_type == "dense"
        assert len(individual_embedding["values"]) == 1024

    def test_embedding_result_is_iterable(self):
        pc = Pinecone()
        embeddings = pc.inference.embed(
            model=EmbedModel.Multilingual_E5_Large,
            inputs=["The quick brown fox jumps over the lazy dog.", "lorem ipsum"],
            parameters={"input_type": "query", "truncate": "END"},
        )
        iter_count = 0
        for embedding in embeddings:
            iter_count += 1
            assert len(embedding.values) == 1024
        assert iter_count == 2

    @pytest.mark.parametrize(
        "model_input,model_output",
        [
            (EmbedModel.Pinecone_Sparse_English_V0, "pinecone-sparse-english-v0"),
            ("pinecone-sparse-english-v0", "pinecone-sparse-english-v0"),
        ],
    )
    def test_create_sparse_embeddings(self, model_input, model_output):
        pc = Pinecone()
        embeddings = pc.inference.embed(
            model=model_input,
            inputs=["The quick brown fox jumps over the lazy dog.", "lorem ipsum"],
            parameters={"input_type": "query", "truncate": "END"},
        )
        assert embeddings.vector_type == "sparse"
        assert embeddings.model == model_output
        assert len(embeddings.data) == 2

    def test_create_embeddings_input_objects(self):
        pc = Pinecone()
        embeddings = pc.inference.embed(
            model=pc.inference.EmbedModel.Multilingual_E5_Large,
            inputs=[
                {"text": "The quick brown fox jumps over the lazy dog."},
                {"text": "lorem ipsum"},
            ],
            parameters={"input_type": "query", "truncate": "END"},
        )
        assert len(embeddings.get("data")) == 2
        assert len(embeddings.get("data")[0]["values"]) == 1024
        assert len(embeddings.get("data")[1]["values"]) == 1024
        assert embeddings.get("model") == "multilingual-e5-large"

    def test_create_embeddings_input_string(self):
        pc = Pinecone()
        embeddings = pc.inference.embed(
            model=EmbedModel.Multilingual_E5_Large,
            inputs="The quick brown fox jumps over the lazy dog.",
            parameters={"input_type": "query", "truncate": "END"},
        )
        assert len(embeddings.get("data")) == 1
        assert len(embeddings.get("data")[0]["values"]) == 1024
        assert embeddings.get("model") == "multilingual-e5-large"

    def test_create_embeddings_invalid_input_empty_list(self):
        pc = Pinecone()
        embedding_model = "multilingual-e5-large"
        with pytest.raises(Exception) as excinfo:
            pc.inference.embed(
                model=embedding_model,
                inputs=[],
                parameters={"input_type": "query", "truncate": "END"},
            )
        assert str(excinfo.value).find("Invalid type") >= 0

    def test_create_embeddings_invalid_input(self):
        pc = Pinecone()
        embedding_model = "multilingual-e5-large"
        with pytest.raises(Exception) as excinfo:
            pc.inference.embed(
                model=embedding_model,
                inputs=[{"INVALID_FIELD": "this should be rejected"}],
                parameters={"input_type": "query", "truncate": "END"},
            )
        assert str(excinfo.value).find("INVALID_FIELD") >= 0

    def test_can_attempt_to_use_unknown_models(self):
        pc = Pinecone()

        # We don't want to reject these requests client side because we want
        # to remain forwards compatible with any new models that become available
        model = "unknown-model"
        with pytest.raises(PineconeApiException) as excinfo:
            pc.inference.embed(
                model=model,
                inputs="The quick brown fox jumps over the lazy dog.",
                parameters={"input_type": "query", "truncate": "END"},
            )
        assert "Model 'unknown-model' not found" in str(excinfo.value)
