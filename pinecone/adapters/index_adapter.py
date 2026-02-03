"""Adapter functions for IndexModel spec resolution.

This module provides adapter functions that handle the complex oneOf schema resolution
for IndexModel spec fields. This isolates the SDK wrapper code from the internal
structure of OpenAPI models and their deserialization logic.

The adapter extracts spec resolution logic from the IndexModel wrapper, making it
easier to support future API format changes (e.g., schema-based dimension/metric).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pinecone.openapi_support.model_utils import deserialize_model

if TYPE_CHECKING:
    from pinecone.adapters.protocols import IndexModelAdapter


def adapt_index_spec(index: "IndexModelAdapter") -> Any:
    """Adapt an IndexModel's spec field, handling oneOf schema resolution.

    The OpenAPI spec for IndexModel.spec is a oneOf union of Serverless, PodBased,
    and BYOC types. The OpenAPI generator's deserialization sometimes fails to properly
    resolve which variant to use, leaving spec as a raw dict. This adapter manually
    detects the correct type and constructs the appropriate wrapper.

    This function handles three spec types:
    - serverless: Contains nested ServerlessSpecResponse with optional ReadCapacity
    - pod: Contains nested PodSpec
    - byoc: Contains nested ByocSpec

    Args:
        index: An IndexModel-like object conforming to IndexModelAdapter protocol.

    Returns:
        The deserialized IndexSpec (Serverless, PodBased, or BYOC), or None if spec
        is not present. Returns Any to satisfy mypy since the actual return types
        are oneOf variants that don't have a common base type accessible here.

    Example:
        >>> spec = adapt_index_spec(index)
        >>> if hasattr(spec, 'serverless'):
        ...     print(f"Cloud: {spec.serverless.cloud}")
    """
    from pinecone.core.openapi.db_control.model.index_spec import IndexSpec

    # Access _data_store directly to avoid OpenAPI model attribute resolution
    spec_value = index._data_store.get("spec")
    if spec_value is None:
        # Fallback to getattr in case spec is stored differently
        spec_value = getattr(index, "spec", None)

    if not isinstance(spec_value, dict):
        # Already an IndexSpec instance or None
        return spec_value

    # Get configuration from the underlying model for proper deserialization
    config = index._configuration
    path_to_item = index._path_to_item

    # Convert to list if needed and append 'spec' to path_to_item for proper error reporting
    if isinstance(path_to_item, (list, tuple)):
        spec_path = list(path_to_item) + ["spec"]
    else:
        spec_path = ["spec"]

    # Manually detect which oneOf schema to use based on discriminator keys
    if "serverless" in spec_value:
        return _adapt_serverless_spec(spec_value, spec_path, config)
    elif "pod" in spec_value:
        return _adapt_pod_spec(spec_value, spec_path, config)
    elif "byoc" in spec_value:
        return _adapt_byoc_spec(spec_value, spec_path, config)
    else:
        # Fallback: try deserialize_model (shouldn't happen with valid API responses)
        return deserialize_model(
            spec_value,
            IndexSpec,
            spec_path,
            check_type=True,
            configuration=config,
            spec_property_naming=False,
        )


def _adapt_serverless_spec(spec_value: dict[str, Any], spec_path: list[str], config: Any) -> Any:
    """Adapt a serverless spec, handling nested ReadCapacity oneOf.

    Args:
        spec_value: Raw spec dict from _data_store
        spec_path: Path to spec in response tree for error reporting
        config: OpenAPI configuration object

    Returns:
        Serverless wrapper instance containing ServerlessSpecResponse
    """
    from pinecone.core.openapi.db_control.model.serverless import Serverless
    from pinecone.core.openapi.db_control.model.serverless_spec_response import (
        ServerlessSpecResponse,
    )
    from pinecone.core.openapi.db_control.model.read_capacity_response import ReadCapacityResponse
    from pinecone.core.openapi.db_control.model.read_capacity_on_demand_spec_response import (
        ReadCapacityOnDemandSpecResponse,
    )
    from pinecone.core.openapi.db_control.model.read_capacity_dedicated_spec_response import (
        ReadCapacityDedicatedSpecResponse,
    )

    # Deserialize the nested serverless dict to ServerlessSpecResponse
    serverless_dict = dict(spec_value["serverless"])

    # Handle nested read_capacity if present (it's also a oneOf with discriminator)
    # Preserve already-deserialized values, only deserialize dicts
    read_capacity_spec = serverless_dict.get("read_capacity")
    if "read_capacity" in serverless_dict and isinstance(serverless_dict["read_capacity"], dict):
        read_capacity_dict = serverless_dict["read_capacity"]
        mode = read_capacity_dict.get("mode")

        # Use discriminator to determine which ReadCapacity spec to use
        if mode == "OnDemand":
            read_capacity_spec = deserialize_model(
                read_capacity_dict,
                ReadCapacityOnDemandSpecResponse,
                spec_path + ["serverless", "read_capacity"],
                check_type=True,
                configuration=config,
                spec_property_naming=False,
            )
        elif mode == "Dedicated":
            read_capacity_spec = deserialize_model(
                read_capacity_dict,
                ReadCapacityDedicatedSpecResponse,
                spec_path + ["serverless", "read_capacity"],
                check_type=True,
                configuration=config,
                spec_property_naming=False,
            )
        else:
            # Fallback to ReadCapacityResponse (should use discriminator)
            read_capacity_spec = deserialize_model(
                read_capacity_dict,
                ReadCapacityResponse,
                spec_path + ["serverless", "read_capacity"],
                check_type=True,
                configuration=config,
                spec_property_naming=False,
            )

    # Create ServerlessSpecResponse with all required and optional fields
    serverless_spec = ServerlessSpecResponse._from_openapi_data(
        cloud=serverless_dict["cloud"],
        region=serverless_dict["region"],
        read_capacity=read_capacity_spec,
        source_collection=serverless_dict.get("source_collection"),
        schema=serverless_dict.get("schema"),
        _check_type=False,
        _path_to_item=spec_path + ["serverless"],
        _configuration=config,
        _spec_property_naming=False,
    )

    # Instantiate Serverless wrapper, which IS the IndexSpec (oneOf union)
    # Note: We use _check_type=False because ServerlessSpecResponse (from GET responses)
    # is compatible with but not identical to ServerlessSpec (used in POST requests)
    return Serverless._new_from_openapi_data(
        serverless=serverless_spec,
        _check_type=False,
        _path_to_item=spec_path,
        _configuration=config,
        _spec_property_naming=False,
    )


def _adapt_pod_spec(spec_value: dict[str, Any], spec_path: list[str], config: Any) -> Any:
    """Adapt a pod-based spec.

    Args:
        spec_value: Raw spec dict from _data_store
        spec_path: Path to spec in response tree for error reporting
        config: OpenAPI configuration object

    Returns:
        PodBased wrapper instance containing PodSpec
    """
    from pinecone.core.openapi.db_control.model.pod_based import PodBased
    from pinecone.core.openapi.db_control.model.pod_spec import PodSpec

    pod_spec = deserialize_model(
        spec_value["pod"],
        PodSpec,
        spec_path + ["pod"],
        check_type=True,
        configuration=config,
        spec_property_naming=False,
    )

    return PodBased._new_from_openapi_data(
        pod=pod_spec,
        _check_type=True,
        _path_to_item=spec_path,
        _configuration=config,
        _spec_property_naming=False,
    )


def _adapt_byoc_spec(spec_value: dict[str, Any], spec_path: list[str], config: Any) -> Any:
    """Adapt a BYOC (Bring Your Own Cloud) spec.

    Args:
        spec_value: Raw spec dict from _data_store
        spec_path: Path to spec in response tree for error reporting
        config: OpenAPI configuration object

    Returns:
        BYOC wrapper instance containing ByocSpec
    """
    from pinecone.core.openapi.db_control.model.byoc import BYOC
    from pinecone.core.openapi.db_control.model.byoc_spec import ByocSpec

    byoc_spec = deserialize_model(
        spec_value["byoc"],
        ByocSpec,
        spec_path + ["byoc"],
        check_type=True,
        configuration=config,
        spec_property_naming=False,
    )

    return BYOC._new_from_openapi_data(
        byoc=byoc_spec,
        _check_type=True,
        _path_to_item=spec_path,
        _configuration=config,
        _spec_property_naming=False,
    )
