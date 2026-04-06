"""Unit tests for AsyncInference namespace — async embed, rerank, and model methods."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.async_client.inference import AsyncInference
from pinecone.models.enums import EmbedModel, RerankModel
from pinecone.models.inference.embed import EmbeddingsList
from pinecone.models.inference.model_list import ModelInfoList
from pinecone.models.inference.models import ModelInfo
from pinecone.models.inference.rerank import RerankResult
from tests.factories import (
    make_embed_response,
    make_model_info,
    make_model_list_response,
    make_rerank_response,
)

BASE_URL = "https://api.test.pinecone.io"


@pytest.fixture()
def config() -> PineconeConfig:
    return PineconeConfig(api_key="test-key", host=BASE_URL)


@pytest.fixture()
def inference(config: PineconeConfig) -> AsyncInference:
    return AsyncInference(config=config)


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_async_embed_success(inference: AsyncInference) -> None:
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = await inference.embed("multilingual-e5-large", ["hello"])

    assert isinstance(result, EmbeddingsList)
    assert result.model == "multilingual-e5-large"
    assert len(result.data) == 1
    assert result.data[0].values == [0.1, 0.2, 0.3]
    assert result.usage.total_tokens == 205
    assert route.called


@respx.mock
@pytest.mark.asyncio
async def test_async_embed_single_string(inference: AsyncInference) -> None:
    route = respx.post(f"{BASE_URL}/embed").mock(
        return_value=httpx.Response(200, json=make_embed_response()),
    )

    result = await inference.embed("multilingual-e5-large", "hello")

    assert isinstance(result, EmbeddingsList)
    import orjson

    body = orjson.loads(route.calls[0].request.content)
    assert body["inputs"] == [{"text": "hello"}]


@pytest.mark.asyncio
async def test_async_embed_empty_raises(inference: AsyncInference) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        await inference.embed("multilingual-e5-large", [])


# ---------------------------------------------------------------------------
# rerank()
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_async_rerank_success(inference: AsyncInference) -> None:
    route = respx.post(f"{BASE_URL}/rerank").mock(
        return_value=httpx.Response(200, json=make_rerank_response()),
    )

    result = await inference.rerank(
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


@pytest.mark.asyncio
async def test_async_rerank_empty_docs_raises(inference: AsyncInference) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        await inference.rerank(
            model="bge-reranker-v2-m3",
            query="test query",
            documents=[],
        )


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_async_list_models(inference: AsyncInference) -> None:
    route = respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=make_model_list_response()),
    )

    result = await inference.list_models()

    assert isinstance(result, ModelInfoList)
    assert len(result) == 2
    assert result.names() == ["multilingual-e5-large", "bge-reranker-v2-m3"]
    assert route.called


# ---------------------------------------------------------------------------
# get_model()
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_async_get_model(inference: AsyncInference) -> None:
    route = respx.get(f"{BASE_URL}/models/multilingual-e5-large").mock(
        return_value=httpx.Response(200, json=make_model_info()),
    )

    result = await inference.get_model(model_name="multilingual-e5-large")

    assert isinstance(result, ModelInfo)
    assert result.model == "multilingual-e5-large"
    assert result.type == "embed"
    assert route.called


# ---------------------------------------------------------------------------
# Lazy property on AsyncPinecone
# ---------------------------------------------------------------------------


def test_async_inference_lazy_property() -> None:
    """Accessing .inference twice returns the same instance."""
    with patch.dict("os.environ", {"PINECONE_API_KEY": "test-key"}):
        from pinecone.async_client.pinecone import AsyncPinecone

        pc = AsyncPinecone(api_key="test-key")
        first = pc.inference
        second = pc.inference
        assert first is second


# ---------------------------------------------------------------------------
# repr and class attributes
# ---------------------------------------------------------------------------


def test_async_inference_repr(inference: AsyncInference) -> None:
    assert repr(inference) == "AsyncInference()"


def test_async_inference_class_attributes() -> None:
    """AsyncInference exposes EmbedModel and RerankModel as class attributes."""
    assert AsyncInference.EmbedModel is EmbedModel
    assert AsyncInference.RerankModel is RerankModel
