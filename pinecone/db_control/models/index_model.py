from pinecone.core.openapi.db_control.model.index_model import IndexModel as OpenAPIIndexModel
from pinecone.core.openapi.db_control.model.index_spec import IndexSpec
import json
from pinecone.utils.repr_overrides import custom_serializer


class IndexModel:
    def __init__(self, index: OpenAPIIndexModel):
        self.index = index
        self._spec_cache = None

    def __str__(self):
        return str(self.index)

    def __getattr__(self, attr):
        if attr == "spec":
            return self._get_spec()
        return getattr(self.index, attr)

    def _get_spec(self):
        if self._spec_cache is not None:
            return self._spec_cache

        # Access _data_store directly to avoid OpenAPI model attribute resolution
        spec_value = self.index._data_store.get("spec")
        if spec_value is None:
            # Fallback to getattr in case spec is stored differently
            spec_value = getattr(self.index, "spec", None)

        if isinstance(spec_value, dict):
            # Use _new_from_openapi_data for proper deserialization of nested dicts
            # Get configuration from the underlying model if available
            config = getattr(self.index, "_configuration", None)
            path_to_item = getattr(self.index, "_path_to_item", ())
            # Unpack dict as kwargs (like deserialize_model does) for proper nested conversion
            kw_args = {
                "_check_type": True,
                "_path_to_item": path_to_item,
                "_configuration": config,
                "_spec_property_naming": False,
            }
            kw_args.update(spec_value)
            self._spec_cache = IndexSpec._new_from_openapi_data(**kw_args)
        elif spec_value is None:
            self._spec_cache = None
        else:
            # Already an IndexSpec instance or some other object
            self._spec_cache = spec_value

        return self._spec_cache

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4, default=custom_serializer)

    def to_dict(self):
        return self.index.to_dict()
