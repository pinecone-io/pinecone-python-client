import pytest
from pinecone import Pinecone
from pinecone.grpc import PineconeGRPC
from pinecone.exceptions import PineconeApiException


class TestInferencePluginRerank:
    def test_rerank(self, api_key):
        pc = Pinecone(api_key=api_key)

        model = "bge-reranker-v2-m3"
        result = pc.inference.rerank(
            model=model,
            query="i love dogs",
            documents=[
                "dogs are pretty cool",
                "everyone loves dogs",
                "I'm a cat person",
            ],
            top_n=1,
            return_documents=True,
        )
        assert len(result.data) == 1
        assert result.data[0].index == 1
        assert result.data[0].document.text == "everyone loves dogs"
        assert result.model == model
        assert isinstance(result.usage.rerank_units, int)
        assert result.usage.rerank_units == 1

    def test_rerank_grpc(self, api_key):
        pc = PineconeGRPC(api_key=api_key)

        model = "bge-reranker-v2-m3"
        result = pc.inference.rerank(
            model=model,
            query="i love dogs",
            documents=[
                "dogs are pretty cool",
                "everyone loves dogs",
                "I'm a cat person",
            ],
            top_n=1,
            return_documents=True,
        )
        assert len(result.data) == 1
        assert result.data[0].index == 1
        assert result.data[0].document.text == "everyone loves dogs"
        assert result.model == model
        assert isinstance(result.usage.rerank_units, int)
        assert result.usage.rerank_units == 1

    def test_rerank_exception(self, api_key):
        pc = Pinecone(api_key=api_key)
        with pytest.raises(PineconeApiException) as e_info:
            pc.inference.rerank(
                model="DOES NOT EXIST",
                query="i love dogs",
                documents=[
                    "dogs are pretty cool",
                    "everyone loves dogs",
                    "I'm a cat person",
                ],
                rank_fields=["custom-field"],
                top_n=1,
                return_documents=True,
            )
        assert e_info.value.status == 404
