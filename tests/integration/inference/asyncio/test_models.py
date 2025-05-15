import pytest
from pinecone import PineconeAsyncio
import logging

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
class TestModels:
    async def test_list_models(self):
        async with PineconeAsyncio() as pc:
            models = await pc.inference.list_models()
            assert len(models) > 0
            logger.info(f"Models[0]: {models[0]}")
            assert models[0].model is not None
            assert models[0].short_description is not None
            assert models[0].type is not None
            assert models[0].supported_parameters is not None
            assert models[0].modality is not None
            assert models[0].max_sequence_length is not None
            assert models[0].max_batch_size is not None
            assert models[0].provider_name is not None

    async def test_list_models_with_type(self):
        async with PineconeAsyncio() as pc:
            models = await pc.inference.list_models(type="embed")
            assert len(models) > 0
            assert models[0].type == "embed"

            models2 = await pc.inference.list_models(type="rerank")
        assert len(models2) > 0
        assert models2[0].type == "rerank"

    async def test_list_models_with_vector_type(self):
        async with PineconeAsyncio() as pc:
            models = await pc.inference.list_models(vector_type="dense")
            assert len(models) > 0
            assert models[0].vector_type == "dense"

            models2 = await pc.inference.list_models(vector_type="sparse")
            assert len(models2) > 0
            assert models2[0].vector_type == "sparse"

    async def test_list_models_with_type_and_vector_type(self):
        async with PineconeAsyncio() as pc:
            models = await pc.inference.list_models(type="embed", vector_type="dense")
            assert len(models) > 0
            assert models[0].type == "embed"
            assert models[0].vector_type == "dense"
