"""Priority-1 smoke test — async inference surface.

Mirror of ``test_inference_sync.py`` against ``AsyncPinecone``.

Punchlist coverage:

- AsyncPinecone.inference.embed / rerank / list_models / get_model
- AsyncPinecone.inference.model.list / model.get
- AsyncPinecone.inference.close
- AsyncPinecone.close, ``async with`` context manager
"""

from __future__ import annotations

import pytest

from pinecone import AsyncPinecone

EMBED_MODEL = "multilingual-e5-large"
RERANK_MODEL = "bge-reranker-v2-m3"


@pytest.mark.smoke
@pytest.mark.anyio
async def test_inference_smoke_async(async_client: AsyncPinecone, api_key: str) -> None:
    """End-to-end async inference surface walkthrough."""
    # embed — list of strings
    result = await async_client.inference.embed(
        model=EMBED_MODEL,
        inputs=["hello", "world"],
        parameters={"input_type": "passage"},
    )
    assert len(result) == 2
    assert result.model == EMBED_MODEL

    # embed — list of dicts
    dict_result = await async_client.inference.embed(
        model=EMBED_MODEL,
        inputs=[{"text": "hi"}, {"text": "there"}],
        parameters={"input_type": "passage"},
    )
    assert len(dict_result) == 2

    # rerank
    rerank = await async_client.inference.rerank(
        model=RERANK_MODEL,
        query="puppies and dogs",
        documents=["a small puppy", "a parked car", "a golden retriever"],
    )
    scores = [item.score for item in rerank.data]
    assert scores == sorted(scores, reverse=True)

    # list_models / get_model
    models = await async_client.inference.list_models()
    assert EMBED_MODEL in models.names()
    embed_info = await async_client.inference.get_model(model=EMBED_MODEL)
    assert embed_info.type == "embed"

    # model.list / model.get aliases
    via_alias = await async_client.inference.model.list()
    assert EMBED_MODEL in via_alias.names()
    rerank_info = await async_client.inference.model.get(RERANK_MODEL)
    assert rerank_info.type == "rerank"

    # explicit close on the inference sub-client
    await async_client.inference.close()

    # async with context manager — tests the context-manager form with a fresh client
    async with AsyncPinecone(api_key=api_key) as ctx_pc:
        ctx_result = await ctx_pc.inference.embed(
            model=EMBED_MODEL,
            inputs=["context-manager smoke check"],
            parameters={"input_type": "passage"},
        )
        assert len(ctx_result) == 1
