import logging
from pinecone import Pinecone

logger = logging.getLogger(__name__)


class TestListModels:
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

    def test_list_models_new_syntax(self):
        pc = Pinecone()
        models = pc.inference.model.list(type="embed", vector_type="dense")
        assert len(models) > 0
        logger.info(f"Models[0]: {models[0]}")
        assert models[0].model is not None
        assert models[0].short_description is not None

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


class TestGetModel:
    def test_get_model(self):
        pc = Pinecone()
        models = pc.inference.list_models()
        first_model = models[0]

        model = pc.inference.get_model(model_name=first_model.model)
        assert model.model == first_model.model
        assert model.short_description == first_model.short_description
        assert model.type == first_model.type
        assert model.supported_parameters == first_model.supported_parameters
        assert model.modality == first_model.modality
        assert model.max_sequence_length == first_model.max_sequence_length
        assert model.max_batch_size == first_model.max_batch_size
        assert model.provider_name == first_model.provider_name

    def test_get_model_new_syntax(self):
        pc = Pinecone()
        models = pc.inference.model.list()
        first_model = models[0]

        model = pc.inference.model.get(model_name=first_model.model)
        assert model.model == first_model.model
        assert model.short_description == first_model.short_description
        assert model.type == first_model.type
        assert model.supported_parameters == first_model.supported_parameters
        assert model.modality == first_model.modality
        assert model.max_sequence_length == first_model.max_sequence_length
