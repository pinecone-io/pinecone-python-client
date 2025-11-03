from pinecone.core.openapi.db_control.model.index_model import IndexModel as OpenAPIIndexModel
import json
from pinecone.utils.repr_overrides import custom_serializer
from pinecone.utils.dict_like import DictLike


class IndexModel:
    def __init__(self, index: OpenAPIIndexModel):
        self.index = index

    def __str__(self):
        return str(self.index)

    def __getattr__(self, attr):
        value = getattr(self.index, attr)
        # If the attribute is 'spec' and it's a dictionary, wrap it in DictLike
        if attr == "spec" and isinstance(value, dict):
            return DictLike(value)
        return value

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4, default=custom_serializer)

    def to_dict(self):
        return self.index.to_dict()
