import pytest
from pinecone import PineconeAsyncio, PineconeApiException, RerankModel, EmbedModel


@pytest.mark.asyncio
class TestEmbedAsyncio:
    @pytest.mark.parametrize(
        "model_input,model_output",
        [
            (EmbedModel.Multilingual_E5_Large, "multilingual-e5-large"),
            ("multilingual-e5-large", "multilingual-e5-large"),
        ],
    )
    async def test_create_embeddings(self, model_input, model_output):
        pc = PineconeAsyncio()
        embeddings = await pc.inference.embed(
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
        assert individual_embedding.vector_type.value == "dense"
        assert len(individual_embedding["values"]) == 1024

        await pc.close()

    async def test_embedding_result_is_iterable(self):
        pc = PineconeAsyncio()
        embeddings = await pc.inference.embed(
            model=EmbedModel.Multilingual_E5_Large,
            inputs=["The quick brown fox jumps over the lazy dog.", "lorem ipsum"],
            parameters={"input_type": "query", "truncate": "END"},
        )
        iter_count = 0
        for embedding in embeddings:
            iter_count += 1
            assert len(embedding.values) == 1024
        assert iter_count == 2
        await pc.close()

    @pytest.mark.parametrize(
        "model_input,model_output",
        [
            (EmbedModel.Pinecone_Sparse_English_V0, "pinecone-sparse-english-v0"),
            ("pinecone-sparse-english-v0", "pinecone-sparse-english-v0"),
        ],
    )
    async def test_create_sparse_embeddings(self, model_input, model_output):
        pc = PineconeAsyncio()
        embeddings = await pc.inference.embed(
            model=model_input,
            inputs=["The quick brown fox jumps over the lazy dog.", "lorem ipsum"],
            parameters={"input_type": "query", "truncate": "END"},
        )
        assert embeddings.vector_type == "sparse"
        assert embeddings.model == model_output
        assert len(embeddings.data) == 2
        await pc.close()

    async def test_create_embeddings_input_objects(self):
        pc = PineconeAsyncio()
        embeddings = await pc.inference.embed(
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
        await pc.close()

    async def test_create_embeddings_input_string(self):
        pc = PineconeAsyncio()
        embeddings = await pc.inference.embed(
            model=EmbedModel.Multilingual_E5_Large,
            inputs="The quick brown fox jumps over the lazy dog.",
            parameters={"input_type": "query", "truncate": "END"},
        )
        assert len(embeddings.get("data")) == 1
        assert len(embeddings.get("data")[0]["values"]) == 1024
        assert embeddings.get("model") == "multilingual-e5-large"
        await pc.close()

    async def test_create_embeddings_invalid_input_empty_list(self):
        pc = PineconeAsyncio()
        embedding_model = "multilingual-e5-large"
        with pytest.raises(Exception) as excinfo:
            await pc.inference.embed(
                model=embedding_model,
                inputs=[],
                parameters={"input_type": "query", "truncate": "END"},
            )
        assert str(excinfo.value).find("Invalid type") >= 0
        await pc.close()

    async def test_create_embeddings_invalid_input(self):
        pc = PineconeAsyncio()
        embedding_model = "multilingual-e5-large"
        with pytest.raises(Exception) as excinfo:
            await pc.inference.embed(
                model=embedding_model,
                inputs=[{"INVALID_FIELD": "this should be rejected"}],
                parameters={"input_type": "query", "truncate": "END"},
            )
        assert str(excinfo.value).find("INVALID_FIELD") >= 0
        await pc.close()

    async def test_can_attempt_to_use_unknown_models(self):
        pc = PineconeAsyncio()

        # We don't want to reject these requests client side because we want
        # to remain forwards compatible with any new models that become available
        model = "unknown-model"
        with pytest.raises(PineconeApiException) as excinfo:
            await pc.inference.embed(
                model=model,
                inputs="The quick brown fox jumps over the lazy dog.",
                parameters={"input_type": "query", "truncate": "END"},
            )
        assert "Model 'unknown-model' not found" in str(excinfo.value)
        await pc.close()


@pytest.mark.asyncio
class TestRerankAsyncio:
    @pytest.mark.parametrize(
        "model_input,model_output",
        [
            (RerankModel.Bge_Reranker_V2_M3, "bge-reranker-v2-m3"),
            ("bge-reranker-v2-m3", "bge-reranker-v2-m3"),
        ],
    )
    async def test_rerank_basic(self, model_input, model_output):
        # Rerank model can be passed as string or enum
        pc = PineconeAsyncio()
        result = await pc.inference.rerank(
            model=model_input,
            query="i love dogs",
            documents=["dogs are pretty cool", "everyone loves dogs", "I'm a cat person"],
            top_n=1,
            return_documents=True,
        )
        assert len(result.data) == 1
        assert result.data[0].index == 1
        assert result.data[0].document.text == "everyone loves dogs"
        assert result.model == model_output
        assert isinstance(result.usage.rerank_units, int)
        assert result.usage.rerank_units == 1
        await pc.close()

    async def test_rerank_basic_document_dicts(self):
        model = "bge-reranker-v2-m3"
        pc = PineconeAsyncio()
        result = await pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query="i love dogs",
            documents=[
                {"id": "123", "text": "dogs are pretty cool"},
                {"id": "789", "text": "I'm a cat person"},
                {"id": "456", "text": "everyone loves dogs"},
            ],
            top_n=1,
            return_documents=True,
        )
        assert len(result.data) == 1
        assert result.data[0].index == 2
        assert result.data[0].document.text == "everyone loves dogs"
        assert result.model == model
        assert isinstance(result.usage.rerank_units, int)
        assert result.usage.rerank_units == 1
        await pc.close()

    async def test_rerank_document_dicts_custom_field(self):
        model = "bge-reranker-v2-m3"
        pc = PineconeAsyncio()
        result = await pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query="i love dogs",
            documents=[
                {"id": "123", "my_field": "dogs are pretty cool"},
                {"id": "456", "my_field": "everyone loves dogs"},
                {"id": "789", "my_field": "I'm a cat person"},
            ],
            rank_fields=["my_field"],
            top_n=1,
            return_documents=True,
        )
        assert len(result.data) == 1
        assert result.data[0].index == 1
        assert result.data[0].document.my_field == "everyone loves dogs"
        assert result.model == model
        assert isinstance(result.usage.rerank_units, int)
        assert result.usage.rerank_units == 1
        await pc.close()

    async def test_rerank_basic_default_top_n(self):
        model = "bge-reranker-v2-m3"
        pc = PineconeAsyncio()
        result = await pc.inference.rerank(
            model=model,
            query="i love dogs",
            documents=["dogs are pretty cool", "everyone loves dogs", "I'm a cat person"],
            return_documents=True,
        )
        assert len(result.data) == 3
        assert result.data[0].index == 1
        assert result.data[0].document.text == "everyone loves dogs"
        assert result.model == model
        assert isinstance(result.usage.rerank_units, int)
        assert result.usage.rerank_units == 1
        await pc.close()

    async def test_rerank_no_return_documents(self):
        pc = PineconeAsyncio()
        model = pc.inference.RerankModel.Bge_Reranker_V2_M3
        result = await pc.inference.rerank(
            model=model,
            query="i love dogs",
            documents=["dogs are pretty cool", "everyone loves dogs", "I'm a cat person"],
            return_documents=False,
        )
        assert len(result.data) == 3
        assert result.data[0].index == 1
        assert not result.data[0].document
        assert result.model == model.value
        assert isinstance(result.usage.rerank_units, int)
        assert result.usage.rerank_units == 1
        await pc.close()

    async def test_rerank_allows_unknown_models_to_be_passed(self):
        pc = PineconeAsyncio()

        # We don't want to reject these requests client side because we want
        # to remain forwards compatible with any new models that become available
        model = "unknown-model"
        with pytest.raises(PineconeApiException) as excinfo:
            await pc.inference.rerank(
                model=model,
                query="i love dogs",
                documents=["dogs are pretty cool", "everyone loves dogs", "I'm a cat person"],
                return_documents=False,
            )
        assert "Model 'unknown-model' not found" in str(excinfo.value)
        await pc.close()
