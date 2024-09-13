from pinecone import Pinecone
from pinecone.grpc import PineconeGRPC


class TestInferencePluginRerank:
    def test_rerank(api_key):
        pc = Pinecone(api_key=api_key)

        model = "bge-reranker-v2-m3"
        result = pc.inference.rerank(
            model=model,
            query="i love dogs",
            documents=["dogs are pretty cool", "everyone loves dogs", "I'm a cat person"],
            top_n=1,
            return_documents=True,
        )
        assert len(result.data) == 1
        assert result.data[0].index == 1
        assert result.data[0].document.text == "everyone loves dogs"
        assert result.model == model
        assert isinstance(result.usage.rerank_units, int)
        assert result.usage.rerank_units == 1

    def test_rerank_grpc(api_key):
        pc = PineconeGRPC(api_key=api_key)

        model = "bge-reranker-v2-m3"
        result = pc.inference.rerank(
            model=model,
            query="i love dogs",
            documents=["dogs are pretty cool", "everyone loves dogs", "I'm a cat person"],
            top_n=1,
            return_documents=True,
        )
        assert len(result.data) == 1
        assert result.data[0].index == 1
        assert result.data[0].document.text == "everyone loves dogs"
        assert result.model == model
        assert isinstance(result.usage.rerank_units, int)
        assert result.usage.rerank_units == 1
