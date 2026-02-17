from pinecone.core.openapi.db_control.model.index_model import IndexModel as OpenAPIIndexModel
from pinecone.core.openapi.db_control.model.index_spec import IndexSpec
from pinecone.core.openapi.db_control.model.serverless import Serverless
from pinecone.core.openapi.db_control.model.serverless_spec_response import ServerlessSpecResponse
from pinecone.core.openapi.db_control.model.read_capacity_response import ReadCapacityResponse
from pinecone.core.openapi.db_control.model.read_capacity_on_demand_spec_response import (
    ReadCapacityOnDemandSpecResponse,
)
from pinecone.core.openapi.db_control.model.read_capacity_dedicated_spec_response import (
    ReadCapacityDedicatedSpecResponse,
)
from pinecone.core.openapi.db_control.model.pod_based import PodBased
from pinecone.core.openapi.db_control.model.pod_spec import PodSpec
from pinecone.core.openapi.db_control.model.byoc import BYOC
from pinecone.core.openapi.db_control.model.byoc_spec_response import ByocSpecResponse
import json
from typing import Any, Dict, List
from pinecone.utils.repr_overrides import custom_serializer
from pinecone.openapi_support.model_utils import deserialize_model


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
            # Manually detect which oneOf schema to use and construct it directly
            # This bypasses the broken oneOf matching logic in deserialize_model
            # Get configuration from the underlying model if available
            config = getattr(self.index, "_configuration", None)
            path_to_item = getattr(self.index, "_path_to_item", ())
            # Convert to list if needed and append 'spec' to path_to_item for proper error reporting
            if isinstance(path_to_item, (list, tuple)):
                spec_path = list(path_to_item) + ["spec"]
            else:
                spec_path = ["spec"]

            # Check which oneOf key exists and construct the appropriate wrapper class
            if "serverless" in spec_value:
                # Deserialize the nested serverless dict to ServerlessSpecResponse
                # (responses use ServerlessSpecResponse, not ServerlessSpec)
                # First, handle nested read_capacity if present (it's also a oneOf with discriminator)
                serverless_dict = dict(spec_value["serverless"])
                if "read_capacity" in serverless_dict and isinstance(
                    serverless_dict["read_capacity"], dict
                ):
                    serverless_dict["read_capacity"] = self._deserialize_read_capacity(
                        serverless_dict["read_capacity"], spec_path + ["serverless"], config
                    )

                serverless_spec = deserialize_model(
                    serverless_dict,
                    ServerlessSpecResponse,
                    spec_path + ["serverless"],
                    check_type=True,
                    configuration=config,
                    spec_property_naming=False,
                )
                # Instantiate Serverless wrapper, which IS the IndexSpec (oneOf union)
                # Note: Disable type checking to prevent TypeError: cannot convert
                # between request/response types (ServerlessSpec vs ServerlessSpecResponse).
                self._spec_cache = Serverless._new_from_openapi_data(
                    serverless=serverless_spec,
                    _check_type=False,
                    _path_to_item=spec_path,
                    _configuration=config,
                    _spec_property_naming=False,
                )
            elif "pod" in spec_value:
                # Deserialize the nested pod dict to PodSpec
                pod_spec = deserialize_model(
                    spec_value["pod"],
                    PodSpec,
                    spec_path + ["pod"],
                    check_type=True,
                    configuration=config,
                    spec_property_naming=False,
                )
                # Instantiate PodBased wrapper, which IS the IndexSpec (oneOf union)
                # Note: Disable type checking similar to serverless and byoc, but in this
                # case there's not an explicit PodSpecResponse model to deserialize to.
                self._spec_cache = PodBased._new_from_openapi_data(
                    pod=pod_spec,
                    _check_type=False,
                    _path_to_item=spec_path,
                    _configuration=config,
                    _spec_property_naming=False,
                )
            elif "byoc" in spec_value:
                # Deserialize the nested byoc dict to ByocSpecResponse
                # (responses use ByocSpecResponse, not ByocSpec)
                # Handle nested read_capacity if present (it's also a oneOf with discriminator)
                byoc_dict = dict(spec_value["byoc"])
                if "read_capacity" in byoc_dict and isinstance(byoc_dict["read_capacity"], dict):
                    byoc_dict["read_capacity"] = self._deserialize_read_capacity(
                        byoc_dict["read_capacity"], spec_path + ["byoc"], config
                    )

                byoc_spec = deserialize_model(
                    byoc_dict,
                    ByocSpecResponse,
                    spec_path + ["byoc"],
                    check_type=True,
                    configuration=config,
                    spec_property_naming=False,
                )
                # Instantiate BYOC wrapper, which IS the IndexSpec (oneOf union)
                # Note: Disable type checking to prevent TypeError: cannot convert
                # between request/response types (ByocSpec vs ByocSpecResponse).
                self._spec_cache = BYOC._new_from_openapi_data(
                    byoc=byoc_spec,
                    _check_type=False,
                    _path_to_item=spec_path,
                    _configuration=config,
                    _spec_property_naming=False,
                )
            else:
                # Fallback: try deserialize_model (shouldn't happen with valid API responses)
                self._spec_cache = deserialize_model(
                    spec_value,
                    IndexSpec,
                    spec_path,
                    check_type=True,
                    configuration=config,
                    spec_property_naming=False,
                )
        elif spec_value is None:
            self._spec_cache = None
        else:
            # Already an IndexSpec instance or some other object
            self._spec_cache = spec_value

        return self._spec_cache

    @staticmethod
    def _deserialize_read_capacity(
        read_capacity_dict: Dict[str, Any], spec_path: List[str], config: Any
    ) -> Any:
        """Deserialize read_capacity dict using discriminator to determine the correct response model.

        This works around broken oneOf matching logic in deserialize_model.

        Args:
            read_capacity_dict: The read_capacity dict from the API response
            spec_path: Path for error reporting
            config: Configuration object

        Returns:
            Deserialized ReadCapacity response model instance
        """
        mode = read_capacity_dict.get("mode")
        # Only enable strict type checking when a configuration object is present.
        # In fixture-based unit tests config is often None and nested dict payloads
        # should still deserialize along this manual oneOf path.
        check_type = config is not None

        if mode == "OnDemand":
            return deserialize_model(
                read_capacity_dict,
                ReadCapacityOnDemandSpecResponse,
                spec_path + ["read_capacity"],
                check_type=check_type,
                configuration=config,
                spec_property_naming=False,
            )
        elif mode == "Dedicated":
            return deserialize_model(
                read_capacity_dict,
                ReadCapacityDedicatedSpecResponse,
                spec_path + ["read_capacity"],
                check_type=check_type,
                configuration=config,
                spec_property_naming=False,
            )
        else:
            # Fallback to ReadCapacityResponse (should use discriminator)
            return deserialize_model(
                read_capacity_dict,
                ReadCapacityResponse,
                spec_path + ["read_capacity"],
                check_type=check_type,
                configuration=config,
                spec_property_naming=False,
            )

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4, default=custom_serializer)

    def to_dict(self):
        return self.index.to_dict()
