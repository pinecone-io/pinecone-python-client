import json
from pinecone.utils.repr_overrides import custom_serializer
from pinecone.core.openapi.inference.model.model_info import ModelInfo as OpenAPIModelInfo


class ModelInfo:
    def __init__(self, model_info: OpenAPIModelInfo):
        self._model_info = model_info

    def __str__(self):
        return str(self._model_info)

    def __getattr__(self, attr):
        return getattr(self._model_info, attr)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4, default=custom_serializer)

    def to_dict(self):
        return self._model_info.to_dict()
