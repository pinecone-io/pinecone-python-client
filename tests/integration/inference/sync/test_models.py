import logging
from pinecone import Pinecone

logger = logging.getLogger(__name__)

#  {
#             "model": "pinecone-rerank-v0",
#             "short_description": "A state of the art reranking model that out-performs competitors on widely accepted benchmarks. It can handle chunks up to 512 tokens (1-2 paragraphs)",
#             "type": "rerank",
#             "supported_parameters": [
#                 {
#                     "parameter": "truncate",
#                     "type": "one_of",
#                     "value_type": "string",
#                     "required": false,
#                     "default": "END",
#                     "allowed_values": [
#                         "END",
#                         "NONE"
#                     ]
#                 }
#             ],
#             "modality": "text",
#             "max_sequence_length": 512,
#             "max_batch_size": 100,
#             "provider_name": "Pinecone"
#         }


class TestModels:
    def test_list_models(self):
        pc = Pinecone()
        models = pc.inference.list_models()
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

    def test_list_models_with_type(self):
        pc = Pinecone()
        models = pc.inference.list_models(type="embed")
        assert len(models) > 0
        assert models[0].type == "embed"

        models2 = pc.inference.list_models(type="rerank")
        assert len(models2) > 0
        assert models2[0].type == "rerank"

    def test_list_models_with_vector_type(self):
        pc = Pinecone()
        models = pc.inference.list_models(vector_type="dense")
        assert len(models) > 0
        assert models[0].vector_type == "dense"

        models2 = pc.inference.list_models(vector_type="sparse")
        assert len(models2) > 0
        assert models2[0].vector_type == "sparse"

    def test_list_models_with_type_and_vector_type(self):
        pc = Pinecone()
        models = pc.inference.list_models(type="embed", vector_type="dense")
        assert len(models) > 0
        assert models[0].type == "embed"
        assert models[0].vector_type == "dense"

    def test_model_can_be_displayed(self):
        # We want to check this, since we're doing some custom
        # shenanigans to the model classes to make them more user
        # friendly. Want to make sure we don't break the basic
        # use case of displaying the model.
        pc = Pinecone()
        models = pc.inference.list_models()
        models.__repr__()  # This should not throw
        models[0].__repr__()  # This should not throw
        models.to_dict()  # This should not throw
        models[0].to_dict()  # This should not throw
        assert True
