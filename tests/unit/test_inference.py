"""Unit tests for Inference namespace — embed and rerank methods."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import INFERENCE_API_VERSION
from pinecone._internal.http_client import HTTPClient
from pinecone.client.inference import Inference
from pinecone.errors.exceptions import ValidationError
from pinecone.models.enums import EmbedModel, RerankModel
from pinecone.models.inference.embed import EmbeddingsList
from pinecone.models.inference.rerank import RerankResult
from tests.factories import make_embed_response, make_rerank_response

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture()
def config() -> PineconeConfig:
    return PineconeConfig(api_key="test-key", host=BASE_URL)


@pytest.fixture()
def inference(config: PineconeConfig) -> Inference:
    return Inference(config=config)


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------


@respx.mock
def test_embed_success(inference: Inference) -> None:
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = inference.embed("multilingual-e5-large", ["hello"])

    assert isinstance(result, EmbeddingsList)
    assert result.model == "multilingual-e5-large"
    assert len(result.data) == 1
    assert result.data[0].values == [0.1, 0.2, 0.3]
    assert result.usage.total_tokens == 205
    assert route.called


@respx.mock
def test_embed_single_string_input(inference: Inference) -> None:
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = inference.embed("multilingual-e5-large", "hello")

    assert isinstance(result, EmbeddingsList)
    # Verify the bare string was normalized to a list of dicts
    request = route.calls[0].request
    import orjson

    body = orjson.loads(request.content)
    assert body["inputs"] == [{"text": "hello"}]


@respx.mock
def test_embed_dict_inputs(inference: Inference) -> None:
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = inference.embed("multilingual-e5-large", [{"text": "hello"}])

    assert isinstance(result, EmbeddingsList)
    import orjson

    body = orjson.loads(route.calls[0].request.content)
    assert body["inputs"] == [{"text": "hello"}]


def test_embed_empty_inputs_raises(inference: Inference) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        inference.embed("multilingual-e5-large", [])


def test_embed_empty_model_raises(inference: Inference) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        inference.embed("", ["hello"])


@respx.mock
def test_embed_with_parameters(inference: Inference) -> None:
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    inference.embed(
        "multilingual-e5-large",
        ["hello"],
        parameters={"input_type": "passage", "truncate": "END"},
    )

    import orjson

    body = orjson.loads(route.calls[0].request.content)
    assert body["parameters"] == {"input_type": "passage", "truncate": "END"}


@respx.mock
def test_embed_with_enum_model(inference: Inference) -> None:
    respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = inference.embed(EmbedModel.MULTILINGUAL_E5_LARGE, ["hello"])

    assert isinstance(result, EmbeddingsList)


# ---------------------------------------------------------------------------
# rerank()
# ---------------------------------------------------------------------------


@respx.mock
def test_rerank_success(inference: Inference) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    result = inference.rerank(
        model="bge-reranker-v2-m3",
        query="Tell me about tech companies",
        documents=[{"text": "Acme Inc. revolutionized tech."}],
        top_n=2,
    )

    assert isinstance(result, RerankResult)
    assert result.model == "bge-reranker-v2-m3"
    assert len(result.data) == 2
    assert result.data[0].score == 0.95
    assert result.usage.rerank_units == 1
    assert route.called


@respx.mock
def test_rerank_string_documents(inference: Inference) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    inference.rerank(
        model="bge-reranker-v2-m3",
        query="test query",
        documents=["doc1", "doc2"],
    )

    import orjson

    body = orjson.loads(route.calls[0].request.content)
    assert body["documents"] == [{"text": "doc1"}, {"text": "doc2"}]


@respx.mock
def test_rerank_default_rank_fields(inference: Inference) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    inference.rerank(
        model="bge-reranker-v2-m3",
        query="test query",
        documents=["doc1"],
    )

    import orjson

    body = orjson.loads(route.calls[0].request.content)
    assert body["rank_fields"] == ["text"]


def test_rerank_empty_documents_raises(inference: Inference) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=[],
        )


def test_rerank_empty_model_raises(inference: Inference) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        inference.rerank(
            model="",
            query="test query",
            documents=["doc1"],
        )


def test_rerank_empty_query_raises(inference: Inference) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        inference.rerank(
            model="bge-reranker-v2-m3",
            query="",
            documents=["doc1"],
        )


@respx.mock
def test_rerank_with_enum_model(inference: Inference) -> None:
    respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    result = inference.rerank(
        model=RerankModel.BGE_RERANKER_V2_M3,
        query="test query",
        documents=["doc1"],
    )

    assert isinstance(result, RerankResult)


# ---------------------------------------------------------------------------
# Lazy property on Pinecone client
# ---------------------------------------------------------------------------


def test_inference_lazy_property() -> None:
    """Accessing .inference twice returns the same instance."""
    with patch.dict("os.environ", {"PINECONE_API_KEY": "test-key"}):
        from pinecone._client import Pinecone

        pc = Pinecone(api_key="test-key")
        first = pc.inference
        second = pc.inference
        assert first is second


def test_inference_repr(inference: Inference) -> None:
    assert repr(inference) == "Inference()"


def test_inference_class_attributes() -> None:
    """Inference exposes EmbedModel and RerankModel as class attributes."""
    assert Inference.EmbedModel is EmbedModel
    assert Inference.RerankModel is RerankModel
