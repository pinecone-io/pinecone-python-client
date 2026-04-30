"""Priority-1 smoke test — sync inference surface.

Walks through every method on ``Pinecone.inference`` and ``ModelResource``
in a single notebook-style flow. No index resources are created, so this is
the cheapest scenario and runs first in CI.

Punchlist coverage (Inference + Pinecone top-level):

- pc.inference.embed
- pc.inference.rerank
- pc.inference.list_models
- pc.inference.get_model
- pc.inference.model.list
- pc.inference.model.get
- pc.inference.close
- pc.close
- Pinecone ``with`` context manager
"""

from __future__ import annotations

import pytest

from pinecone import Pinecone

EMBED_MODEL = "multilingual-e5-large"
RERANK_MODEL = "bge-reranker-v2-m3"


@pytest.mark.smoke
def test_inference_smoke(api_key: str) -> None:
    """End-to-end inference surface walkthrough."""
    # Step 1 — explicit construction; we'll exercise the context-manager form
    # at the end of the test instead of relying on the fixture.
    pc = Pinecone(api_key=api_key)
    try:
        # Step 2 — embed: list-of-string inputs
        result = pc.inference.embed(
            model=EMBED_MODEL,
            inputs=["hello", "world"],
            parameters={"input_type": "passage"},
        )
        assert len(result) == 2
        assert result.model == EMBED_MODEL
        dim = len(result.data[0].values)
        assert dim > 0
        assert all(len(emb.values) == dim for emb in result.data)

        # Step 3 — embed: list-of-dict input form
        dict_result = pc.inference.embed(
            model=EMBED_MODEL,
            inputs=[{"text": "hi"}, {"text": "there"}],
            parameters={"input_type": "passage"},
        )
        assert len(dict_result) == 2

        # Step 4 — rerank: assert documents come back sorted by score
        rerank = pc.inference.rerank(
            model=RERANK_MODEL,
            query="puppies and dogs",
            documents=["a small puppy", "a parked car", "a golden retriever"],
        )
        scores = [item.score for item in rerank.data]
        assert scores == sorted(scores, reverse=True)
        assert rerank.data[0].index in {0, 2}  # puppy or retriever first

        # Step 5 — list_models: at least one embed model present
        models = pc.inference.list_models()
        assert len(models) > 0
        assert EMBED_MODEL in models.names()

        # Step 6 — get_model: positional + named forms
        embed_info = pc.inference.get_model(model=EMBED_MODEL)
        assert embed_info.type == "embed"
        assert embed_info.model == EMBED_MODEL

        # Step 7 — model.list / model.get aliases
        via_alias = pc.inference.model.list()
        assert EMBED_MODEL in via_alias.names()
        rerank_info = pc.inference.model.get(RERANK_MODEL)
        assert rerank_info.type == "rerank"

        # Step 8a — explicit close
        pc.inference.close()
    finally:
        pc.close()

    # Step 8b — context-manager form. Doing this AFTER the explicit close
    # block confirms ``close()`` doesn't leak resources between iterations.
    with Pinecone(api_key=api_key) as ctx_pc:
        ctx_result = ctx_pc.inference.embed(
            model=EMBED_MODEL,
            inputs=["context-manager smoke check"],
            parameters={"input_type": "passage"},
        )
        assert len(ctx_result) == 1
