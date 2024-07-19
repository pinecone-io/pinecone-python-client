from pinecone import Pinecone
from pinecone.grpc import PineconeGRPC


class TestInferencePlugin:
    def test_embed(self, api_key):
        pc = Pinecone(api_key=api_key)

        embedding_model = "multilingual-e5-large"
        embeddings = pc.inference.embed(
            model=embedding_model,
            inputs=["The quick brown fox jumps over the lazy dog.", "lorem ipsum"],
            parameters={"input_type": "query", "truncate": "END"},
        )

        assert len(embeddings.get("data")) == 2
        assert len(embeddings.get("data")[0]["values"]) == 1024
        assert len(embeddings.get("data")[1]["values"]) == 1024
        assert embeddings.get("model") == embedding_model

    def test_embed_grpc(self, api_key):
        pc = PineconeGRPC(api_key=api_key)

        embedding_model = "multilingual-e5-large"
        embeddings = pc.inference.embed(
            model=embedding_model,
            inputs=["The quick brown fox jumps over the lazy dog.", "lorem ipsum"],
            parameters={"input_type": "query", "truncate": "END"},
        )

        assert len(embeddings.get("data")) == 2
        assert len(embeddings.get("data")[0]["values"]) == 1024
        assert len(embeddings.get("data")[1]["values"]) == 1024
        assert embeddings.get("model") == embedding_model
