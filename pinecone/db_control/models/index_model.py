from pinecone.core.openapi.db_control.model.index_model import IndexModel as OpenAPIIndexModel
import json
from typing import Any
from pinecone.utils.repr_overrides import custom_serializer
from pinecone.adapters import adapt_index_spec


class IndexModel:
    def __init__(self, index: OpenAPIIndexModel):
        self.index = index
        self._spec_cache: Any = None

    def __str__(self):
        return str(self.index)

    def __getattr__(self, attr):
        if attr == "spec":
            return self._get_spec()
        return getattr(self.index, attr)

    def _get_spec(self):
        """Get the index spec, using adapter for oneOf schema resolution.

        Delegates to adapt_index_spec() which handles the complex logic of
        deserializing serverless/pod/byoc spec variants.
        """
        if self._spec_cache is not None:
            return self._spec_cache

        self._spec_cache = adapt_index_spec(self.index)
        return self._spec_cache

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4, default=custom_serializer)

    def to_dict(self):
        return self.index.to_dict()
