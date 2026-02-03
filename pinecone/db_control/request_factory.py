from __future__ import annotations

import logging
from typing import Dict, Any, TYPE_CHECKING
from enum import Enum

from pinecone.utils import parse_non_empty_args, convert_enum_to_string

from pinecone.core.openapi.db_control.model.create_collection_request import CreateCollectionRequest
from pinecone.core.openapi.db_control.model.create_index_for_model_request import (
    CreateIndexForModelRequest,
)
from pinecone.core.openapi.db_control.model.create_index_for_model_request_embed import (
    CreateIndexForModelRequestEmbed,
)
from pinecone.core.openapi.db_control.model.create_index_request import CreateIndexRequest
from pinecone.core.openapi.db_control.model.configure_index_request import ConfigureIndexRequest
from pinecone.core.openapi.db_control.model.index_tags import IndexTags
from pinecone.core.openapi.db_control.model.read_capacity_on_demand_spec import (
    ReadCapacityOnDemandSpec,
)
from pinecone.core.openapi.db_control.model.read_capacity_dedicated_spec import (
    ReadCapacityDedicatedSpec,
)
from pinecone.core.openapi.db_control.model.read_capacity_dedicated_spec_response_scaling import (
    ReadCapacityDedicatedSpecResponseScaling,
)
from pinecone.core.openapi.db_control.model.schema import Schema as OpenAPISchema
from pinecone.core.openapi.db_control.model.schema_fields import SchemaFields as OpenAPISchemaFields
from pinecone.core.openapi.db_control.model.create_index_from_backup_request import (
    CreateIndexFromBackupRequest,
)
from pinecone.db_control.models import ServerlessSpec, PodSpec, ByocSpec, IndexModel, IndexEmbed
from pinecone.db_control.models.deployment import (
    ServerlessDeployment,
    PodDeployment,
    ByocDeployment,
)
from pinecone.db_control.models.schema_fields import DenseVectorField, SparseVectorField

from pinecone.db_control.enums import (
    Metric,
    VectorType,
    DeletionProtection,
    PodType,
    CloudProvider,
    AwsRegion,
    GcpRegion,
    AzureRegion,
)
from .types import CreateIndexForModelEmbedTypedDict, ConfigureIndexEmbed

if TYPE_CHECKING:
    from pinecone.db_control.models.serverless_spec import (
        ReadCapacityDict,
        MetadataSchemaFieldConfig,
    )
    from pinecone.core.openapi.db_control.model.read_capacity import ReadCapacity

logger = logging.getLogger(__name__)
""" :meta private: """


class PineconeDBControlRequestFactory:
    """
    :meta private:

    This class facilitates translating user inputs into request objects.
    """

    @staticmethod
    def __parse_tags(tags: dict[str, str] | None) -> IndexTags:
        from typing import cast

        if tags is None:
            result = IndexTags()
            return cast(IndexTags, result)
        else:
            result = IndexTags(**tags)
            return cast(IndexTags, result)

    @staticmethod
    def __parse_deletion_protection(deletion_protection: DeletionProtection | str) -> str:
        deletion_protection = convert_enum_to_string(deletion_protection)
        if deletion_protection in ["enabled", "disabled"]:
            return deletion_protection
        else:
            raise ValueError("deletion_protection must be either 'enabled' or 'disabled'")

    @staticmethod
    def __parse_read_capacity(
        read_capacity: (
            "ReadCapacityDict"
            | "ReadCapacity"
            | "ReadCapacityOnDemandSpec"
            | "ReadCapacityDedicatedSpec"
        ),
    ) -> ReadCapacityOnDemandSpec | ReadCapacityDedicatedSpec | "ReadCapacity":
        """Parse read_capacity dict into appropriate ReadCapacity model instance.

        :param read_capacity: Dict with read capacity configuration or existing ReadCapacity model instance
        :return: ReadCapacityOnDemandSpec, ReadCapacityDedicatedSpec, or existing model instance
        """
        from typing import cast

        if isinstance(read_capacity, dict):
            mode = read_capacity.get("mode", "OnDemand")
            if mode == "OnDemand":
                result = ReadCapacityOnDemandSpec(mode="OnDemand")
                return cast(ReadCapacityOnDemandSpec, result)
            elif mode == "Dedicated":
                dedicated_dict: dict[str, Any] = read_capacity.get("dedicated", {})  # type: ignore
                # Alpha API structure: node_type and scaling are top-level on ReadCapacityDedicatedSpec
                if "node_type" not in dedicated_dict or dedicated_dict.get("node_type") is None:
                    raise ValueError(
                        "node_type is required when using Dedicated read capacity mode. "
                        "Please specify 'node_type' (e.g., 't1' or 'b1') in the 'dedicated' configuration."
                    )
                node_type = dedicated_dict["node_type"]

                # Handle scaling configuration
                scaling_strategy = dedicated_dict.get("scaling", "Manual")
                manual_dict = dedicated_dict.get("manual", {})
                # Validate manual_dict type
                if manual_dict and not isinstance(manual_dict, dict):
                    raise ValueError(
                        "The 'manual' field must be a dictionary with 'shards' and 'replicas' keys."
                    )
                replicas = manual_dict.get("replicas", 1) if manual_dict else 1
                shards = manual_dict.get("shards", 1) if manual_dict else 1

                # Create the scaling object with the alpha API structure
                scaling_obj = ReadCapacityDedicatedSpecResponseScaling(
                    strategy=scaling_strategy, replicas=replicas, shards=shards
                )

                result = ReadCapacityDedicatedSpec(
                    mode="Dedicated", node_type=node_type, scaling=scaling_obj
                )
                return cast(ReadCapacityDedicatedSpec, result)
            else:
                # Fallback: let OpenAPI handle it
                from typing import cast

                return cast(
                    ReadCapacityOnDemandSpec | ReadCapacityDedicatedSpec | "ReadCapacity",
                    read_capacity,
                )
        else:
            # Already a ReadCapacity model instance
            return read_capacity

    @staticmethod
    def __schema_dict_to_openapi_schema(schema_dict: dict[str, Any]) -> OpenAPISchema:
        """Convert a schema dict to an OpenAPI Schema object.

        :param schema_dict: Dict with 'fields' key containing field configurations.
        :returns: OpenAPI Schema object.
        """
        schema_fields: dict[str, OpenAPISchemaFields] = {}
        for field_name, field_config in schema_dict.get("fields", {}).items():
            if isinstance(field_config, dict):
                field_type = field_config.get("type", "string")
                schema_fields[field_name] = OpenAPISchemaFields(
                    type=field_type,
                    **{k: v for k, v in field_config.items() if k != "type"},
                    _check_type=False,
                )
        return OpenAPISchema(fields=schema_fields, _check_type=False)

    @staticmethod
    def __parse_schema(
        schema: (
            dict[
                str, "MetadataSchemaFieldConfig"
            ]  # Direct field mapping: {field_name: {filterable: bool}}
            | dict[
                str, dict[str, Any]
            ]  # Dict with "fields" wrapper: {"fields": {field_name: {...}}, ...}
            | OpenAPISchema  # OpenAPI model instance
        ),
    ) -> OpenAPISchema:
        """Parse schema dict into Schema instance.

        :param schema: Dict with schema configuration (either {field_name: {type: str, ...}} or
            {"fields": {field_name: {type: str, ...}}, ...}) or existing Schema instance
        :return: Schema instance
        """
        if isinstance(schema, dict):
            schema_kwargs: dict[str, Any] = {}
            # Handle two formats:
            # 1. {field_name: {type: str, ...}} - direct field mapping
            # 2. {"fields": {field_name: {type: str, ...}}, ...} - with fields wrapper
            if "fields" in schema:
                # Format 2: has fields wrapper
                fields = {}
                for field_name, field_config in schema["fields"].items():
                    if isinstance(field_config, dict):
                        # SchemaFields requires 'type' as a required field
                        field_type = field_config.get("type", "string")
                        fields[field_name] = OpenAPISchemaFields(
                            type=field_type,
                            **{k: v for k, v in field_config.items() if k != "type"},
                        )
                    else:
                        # If not a dict, create with default type=string
                        fields[field_name] = OpenAPISchemaFields(type="string")
                schema_kwargs["fields"] = fields

                # Pass through any other fields in schema_dict to allow future API fields
                for key, value in schema.items():
                    if key != "fields":
                        schema_kwargs[key] = value
            else:
                # Format 1: direct field mapping
                # All items in schema are treated as field_name: field_config pairs
                fields = {}
                for field_name, field_config in schema.items():
                    if isinstance(field_config, dict):
                        # SchemaFields requires 'type' as a required field
                        field_type = field_config.get("type", "string")
                        fields[field_name] = OpenAPISchemaFields(
                            type=field_type,
                            **{k: v for k, v in field_config.items() if k != "type"},
                        )
                    else:
                        # If not a dict, create with default type=string
                        fields[field_name] = OpenAPISchemaFields(type="string")
                # Ensure fields is always set, even if empty
                schema_kwargs["fields"] = fields

            # Validate that fields is present before constructing Schema
            if "fields" not in schema_kwargs:
                raise ValueError(
                    "Schema dict must contain field definitions. "
                    "Either provide a 'fields' key with field configurations, "
                    "or provide field_name: field_config pairs directly."
                )

            from typing import cast

            result = OpenAPISchema(**schema_kwargs)
            return cast(OpenAPISchema, result)
        else:
            # Already a Schema instance
            return schema

    @staticmethod
    def _translate_legacy_request(
        spec: Dict | ServerlessSpec | PodSpec | ByocSpec,
        dimension: int | None = None,
        metric: (Metric | str) | None = None,
        vector_type: (VectorType | str) | None = VectorType.DENSE,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Translate legacy spec-based request to deployment + schema format.

        This method converts legacy index creation parameters (spec, dimension, metric,
        vector_type) to the new API format using deployment and schema structures.

        :param spec: The legacy spec (ServerlessSpec, PodSpec, ByocSpec, or dict).
        :param dimension: The vector dimension (for dense vectors).
        :param metric: The distance metric (cosine, euclidean, dotproduct).
        :param vector_type: The vector type (dense or sparse).
        :returns: A tuple of (deployment_dict, schema_dict) for the new API format.

        **Translation Mappings:**

        * `ServerlessSpec(cloud, region)` → `deployment` with `deployment_type="serverless"`
        * `PodSpec(environment, ...)` → `deployment` with `deployment_type="pod"`
        * `ByocSpec(environment)` → `deployment` with `deployment_type="byoc"`
        * `dimension` + `metric` + `vector_type="dense"` → `schema.fields._values` (dense_vector)
        * `vector_type="sparse"` → `schema.fields._sparse_values` (sparse_vector)

        Example::

            deployment, schema = PineconeDBControlRequestFactory._translate_legacy_request(
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                dimension=1536,
                metric="cosine",
                vector_type="dense"
            )
            # Returns:
            # (
            #     {"deployment_type": "serverless", "cloud": "aws", "region": "us-east-1"},
            #     {"fields": {"_values": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}}
            # )
        """
        # Convert metric to string if it's an enum
        if metric is not None:
            metric = convert_enum_to_string(metric)
        if vector_type is not None:
            vector_type = convert_enum_to_string(vector_type)

        # Translate spec to deployment
        deployment_dict: dict[str, Any]
        deployment: ServerlessDeployment | PodDeployment | ByocDeployment
        if isinstance(spec, dict):
            if "serverless" in spec:
                serverless_spec = spec["serverless"]
                # Convert enum values to strings for consistency
                cloud = convert_enum_to_string(serverless_spec.get("cloud", ""))
                region = convert_enum_to_string(serverless_spec.get("region", ""))
                deployment = ServerlessDeployment(cloud=cloud, region=region)
                deployment_dict = deployment.to_dict()
            elif "pod" in spec:
                pod_spec = spec["pod"]
                # Convert enum values to strings for consistency with __parse_index_spec
                environment = convert_enum_to_string(pod_spec.get("environment", ""))
                pod_type = convert_enum_to_string(pod_spec.get("pod_type", "p1.x1"))
                deployment = PodDeployment(
                    environment=environment,
                    pod_type=pod_type,
                    replicas=pod_spec.get("replicas", 1),
                    shards=pod_spec.get("shards", 1),
                    pods=pod_spec.get("pods"),
                )
                deployment_dict = deployment.to_dict()
            elif "byoc" in spec:
                byoc_spec = spec["byoc"]
                # Convert enum values to strings for consistency with __parse_index_spec
                environment = convert_enum_to_string(byoc_spec.get("environment", ""))
                deployment = ByocDeployment(environment=environment)
                deployment_dict = deployment.to_dict()
            else:
                raise ValueError("spec must contain either 'serverless', 'pod', or 'byoc' key")
        elif isinstance(spec, ServerlessSpec):
            deployment = ServerlessDeployment(cloud=spec.cloud, region=spec.region)
            deployment_dict = deployment.to_dict()
        elif isinstance(spec, PodSpec):
            # PodDeployment requires pod_type, but PodSpec defaults to "p1.x1"
            pod_type = spec.pod_type if spec.pod_type is not None else "p1.x1"
            # Use explicit None check to preserve 0 values (consistent with dict handling)
            replicas = spec.replicas if spec.replicas is not None else 1
            shards = spec.shards if spec.shards is not None else 1
            deployment = PodDeployment(
                environment=spec.environment,
                pod_type=pod_type,
                replicas=replicas,
                shards=shards,
                pods=spec.pods,
            )
            deployment_dict = deployment.to_dict()
        elif isinstance(spec, ByocSpec):
            deployment = ByocDeployment(environment=spec.environment)
            deployment_dict = deployment.to_dict()
        else:
            raise TypeError("spec must be of type dict, ServerlessSpec, PodSpec, or ByocSpec")

        # Translate dimension/metric/vector_type to schema
        schema_dict: dict[str, Any] = {"fields": {}}
        if vector_type == VectorType.SPARSE.value:
            # Sparse vector: use _sparse_values field
            if metric is None:
                metric = "dotproduct"  # Default for sparse vectors
            sparse_field = SparseVectorField(metric=metric)
            schema_dict["fields"]["_sparse_values"] = sparse_field.to_dict()
        elif vector_type == VectorType.DENSE.value:
            # Dense vector: use _values field
            if dimension is None:
                raise ValueError("dimension is required for dense vector indexes")
            if metric is None:
                metric = Metric.COSINE.value  # Default for dense vectors
            dense_field = DenseVectorField(dimension=dimension, metric=metric)
            schema_dict["fields"]["_values"] = dense_field.to_dict()
        elif vector_type is not None:
            # Invalid vector_type value - raise error instead of silently returning empty schema
            raise ValueError(
                f"Invalid vector_type: '{vector_type}'. Must be '{VectorType.DENSE.value}' or '{VectorType.SPARSE.value}'"
            )
        # If vector_type is None, return empty schema fields (no vector index)

        return deployment_dict, schema_dict

    @staticmethod
    def _translate_embed_to_semantic_text(
        cloud: CloudProvider | str,
        region: AwsRegion | GcpRegion | AzureRegion | str,
        embed: IndexEmbed | CreateIndexForModelEmbedTypedDict,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Translate embed-based request to deployment + schema format.

        This method converts `create_index_for_model` parameters (cloud, region, embed)
        to the new API format using deployment and schema structures with a semantic_text
        field type.

        :param cloud: The cloud provider (aws, gcp, azure).
        :param region: The cloud region.
        :param embed: The IndexEmbed configuration or equivalent dict.
        :returns: A tuple of (deployment_dict, schema_dict) for the new API format.

        **Translation Example:**

        * Input: ``cloud="aws"``, ``region="us-east-1"``,
          ``embed=IndexEmbed(model="multilingual-e5-large", metric="cosine", field_map={"text": "synopsis"})``
        * Output deployment: ``{"deployment_type": "serverless", "cloud": "aws", "region": "us-east-1"}``
        * Output schema: ``{"fields": {"synopsis": {"type": "semantic_text", "model": "multilingual-e5-large", ...}}}``

        Example::

            deployment, schema = PineconeDBControlRequestFactory._translate_embed_to_semantic_text(
                cloud="aws",
                region="us-east-1",
                embed=IndexEmbed(
                    model="multilingual-e5-large",
                    metric="cosine",
                    field_map={"text": "synopsis"}
                )
            )
        """
        # Convert enum values to strings
        cloud = convert_enum_to_string(cloud)
        region = convert_enum_to_string(region)

        # Create ServerlessDeployment for cloud/region
        deployment = ServerlessDeployment(cloud=cloud, region=region)
        deployment_dict = deployment.to_dict()

        # Parse embed configuration
        model: str
        metric: str | None
        field_map: dict[str, str]
        read_parameters: dict[str, Any] | None
        write_parameters: dict[str, Any] | None

        if isinstance(embed, IndexEmbed):
            model = embed.model
            metric = embed.metric
            field_map = embed.field_map
            read_parameters = embed.read_parameters
            write_parameters = embed.write_parameters
        else:
            # Dict-based embed
            raw_model = embed.get("model")
            if raw_model is None:
                raise ValueError("model is required in embed")
            model = convert_enum_to_string(raw_model)
            raw_metric = embed.get("metric")
            metric = convert_enum_to_string(raw_metric) if raw_metric is not None else None
            raw_field_map = embed.get("field_map")
            if raw_field_map is None:
                raise ValueError("field_map is required in embed")
            field_map = raw_field_map
            read_parameters = embed.get("read_parameters")
            write_parameters = embed.get("write_parameters")

        # Extract field name from field_map values
        # field_map is like {"text": "synopsis"} where "synopsis" is the target field name
        if not field_map:
            raise ValueError("field_map must contain at least one mapping")

        # Build schema with semantic_text fields
        schema_dict: dict[str, Any] = {"fields": {}}

        for _, target_field in field_map.items():
            # Build the semantic_text field configuration
            field_config: dict[str, Any] = {"type": "semantic_text", "model": model}

            # Include metric if provided
            if metric is not None:
                field_config["metric"] = convert_enum_to_string(metric)

            # Apply default read_parameters if not provided or empty
            # Use dict() to create a copy to avoid shared references across fields
            if read_parameters:
                field_config["read_parameters"] = dict(read_parameters)
            else:
                field_config["read_parameters"] = {"input_type": "query"}

            # Apply default write_parameters if not provided or empty
            # Use dict() to create a copy to avoid shared references across fields
            if write_parameters:
                field_config["write_parameters"] = dict(write_parameters)
            else:
                field_config["write_parameters"] = {"input_type": "passage"}

            schema_dict["fields"][target_field] = field_config

        return deployment_dict, schema_dict

    @staticmethod
    def _serialize_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """Serialize schema dict, converting field objects to dicts.

        :param schema: Schema dict with field name to field config mappings.
        :returns: Schema dict with all field objects serialized.
        """
        serialized: dict[str, Any] = {"fields": {}}
        for field_name, field_config in schema.items():
            if hasattr(field_config, "to_dict"):
                # Field type object (TextField, DenseVectorField, etc.)
                serialized["fields"][field_name] = field_config.to_dict()
            elif isinstance(field_config, dict):
                # Already a dict
                serialized["fields"][field_name] = field_config
            else:
                raise TypeError(
                    f"Invalid schema field type for '{field_name}': {type(field_config)}. "
                    "Expected a field type class or dict."
                )
        return serialized

    @staticmethod
    def create_index_with_schema_request(
        name: str,
        schema: dict[str, Any],
        deployment: ServerlessDeployment | PodDeployment | ByocDeployment | None = None,
        deletion_protection: (DeletionProtection | str) | None = DeletionProtection.DISABLED,
        tags: dict[str, str] | None = None,
    ) -> CreateIndexRequest:
        """Create an index request using schema and deployment format.

        This method creates an index request using the new schema-based API format
        rather than the legacy spec-based format.

        :param name: The name of the index.
        :param schema: A dict mapping field names to field configurations.
        :param deployment: The deployment configuration. Defaults to serverless aws/us-east-1.
        :param deletion_protection: Whether to enable deletion protection.
        :param tags: Optional tags for the index.
        :returns: A CreateIndexRequest object for the API.
        """
        if deletion_protection is not None:
            dp = PineconeDBControlRequestFactory.__parse_deletion_protection(deletion_protection)
        else:
            dp = None

        tags_obj = PineconeDBControlRequestFactory.__parse_tags(tags)

        # Default deployment to aws/us-east-1 serverless
        if deployment is None:
            deployment = ServerlessDeployment(cloud="aws", region="us-east-1")

        deployment_dict = deployment.to_dict()
        schema_dict = PineconeDBControlRequestFactory._serialize_schema(schema)
        schema_obj = PineconeDBControlRequestFactory.__schema_dict_to_openapi_schema(schema_dict)

        args = parse_non_empty_args(
            [
                ("name", name),
                ("deployment", deployment_dict),
                ("schema", schema_obj),
                ("deletion_protection", dp),
                ("tags", tags_obj),
            ]
        )

        from typing import cast

        result = CreateIndexRequest(**args, _check_type=False)
        return cast(CreateIndexRequest, result)

    @staticmethod
    def create_index_request(
        name: str,
        spec: Dict | ServerlessSpec | PodSpec | ByocSpec,
        dimension: int | None = None,
        metric: (Metric | str) | None = Metric.COSINE,
        deletion_protection: (DeletionProtection | str) | None = DeletionProtection.DISABLED,
        vector_type: (VectorType | str) | None = VectorType.DENSE,
        tags: dict[str, str] | None = None,
    ) -> CreateIndexRequest:
        if metric is not None:
            metric = convert_enum_to_string(metric)
        if vector_type is not None:
            vector_type = convert_enum_to_string(vector_type)
        if deletion_protection is not None:
            dp = PineconeDBControlRequestFactory.__parse_deletion_protection(deletion_protection)
        else:
            dp = None

        tags_obj = PineconeDBControlRequestFactory.__parse_tags(tags)

        if vector_type == VectorType.SPARSE.value and dimension is not None:
            raise ValueError("dimension should not be specified for sparse indexes")

        # Translate legacy spec/dimension/metric to deployment + schema format for alpha API
        deployment_dict, schema_dict = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, dimension=dimension, metric=metric, vector_type=vector_type
        )
        schema_obj = PineconeDBControlRequestFactory.__schema_dict_to_openapi_schema(schema_dict)

        # Deployment dict is passed directly - OpenAPI model accepts dicts with _check_type=False
        args = parse_non_empty_args(
            [
                ("name", name),
                ("schema", schema_obj),
                ("deployment", deployment_dict),
                ("deletion_protection", dp),
                ("tags", tags_obj),
            ]
        )

        from typing import cast

        result = CreateIndexRequest(**args, _check_type=False)
        return cast(CreateIndexRequest, result)

    @staticmethod
    def create_index_for_model_request(
        name: str,
        cloud: CloudProvider | str,
        region: AwsRegion | GcpRegion | AzureRegion | str,
        embed: IndexEmbed | CreateIndexForModelEmbedTypedDict,
        tags: dict[str, str] | None = None,
        deletion_protection: (DeletionProtection | str) | None = DeletionProtection.DISABLED,
        read_capacity: (
            "ReadCapacityDict"
            | "ReadCapacity"
            | ReadCapacityOnDemandSpec
            | ReadCapacityDedicatedSpec
        )
        | None = None,
        schema: (
            dict[
                str, "MetadataSchemaFieldConfig"
            ]  # Direct field mapping: {field_name: {filterable: bool}}
            | dict[
                str, dict[str, Any]
            ]  # Dict with "fields" wrapper: {"fields": {field_name: {...}}, ...}
            | OpenAPISchema  # OpenAPI model instance
        )
        | None = None,
    ) -> CreateIndexForModelRequest:
        cloud = convert_enum_to_string(cloud)
        region = convert_enum_to_string(region)
        if deletion_protection is not None:
            dp = PineconeDBControlRequestFactory.__parse_deletion_protection(deletion_protection)
        else:
            dp = None
        tags_obj = PineconeDBControlRequestFactory.__parse_tags(tags)

        if isinstance(embed, IndexEmbed):
            parsed_embed = embed.as_dict()
        else:
            # if dict, we need to parse enum values, if any, to string
            # and verify required fields are present
            required_fields = ["model", "field_map"]
            for field in required_fields:
                if field not in embed:
                    raise ValueError(f"{field} is required in embed")
            parsed_embed = {}
            for key, value in embed.items():
                if isinstance(value, Enum):
                    parsed_embed[key] = convert_enum_to_string(value)
                else:
                    parsed_embed[key] = value

        # Parse read_capacity if provided
        parsed_read_capacity = None
        if read_capacity is not None:
            parsed_read_capacity = PineconeDBControlRequestFactory.__parse_read_capacity(
                read_capacity
            )

        # Parse schema if provided
        parsed_schema = None
        if schema is not None:
            parsed_schema = PineconeDBControlRequestFactory.__parse_schema(schema)

        args = parse_non_empty_args(
            [
                ("name", name),
                ("cloud", cloud),
                ("region", region),
                ("embed", CreateIndexForModelRequestEmbed(**parsed_embed)),
                ("deletion_protection", dp),
                ("tags", tags_obj),
                ("read_capacity", parsed_read_capacity),
                ("schema", parsed_schema),
            ]
        )

        from typing import cast

        result = CreateIndexForModelRequest(**args)
        return cast(CreateIndexForModelRequest, result)

    @staticmethod
    def create_index_from_backup_request(
        name: str,
        deletion_protection: (DeletionProtection | str) | None = DeletionProtection.DISABLED,
        tags: dict[str, str] | None = None,
    ) -> CreateIndexFromBackupRequest:
        if deletion_protection is not None:
            dp = PineconeDBControlRequestFactory.__parse_deletion_protection(deletion_protection)
        else:
            dp = None

        tags_obj = PineconeDBControlRequestFactory.__parse_tags(tags)

        from typing import cast

        result = CreateIndexFromBackupRequest(name=name, deletion_protection=dp, tags=tags_obj)
        return cast(CreateIndexFromBackupRequest, result)

    @staticmethod
    def configure_index_request(
        description: IndexModel,
        replicas: int | None = None,
        pod_type: (PodType | str) | None = None,
        deletion_protection: (DeletionProtection | str) | None = None,
        tags: dict[str, str] | None = None,
        embed: (ConfigureIndexEmbed | Dict) | None = None,
        read_capacity: (
            "ReadCapacityDict"
            | "ReadCapacity"
            | ReadCapacityOnDemandSpec
            | ReadCapacityDedicatedSpec
        )
        | None = None,
    ) -> ConfigureIndexRequest:
        if deletion_protection is None:
            dp = description.deletion_protection
        elif isinstance(deletion_protection, DeletionProtection):
            dp = deletion_protection.value
        elif deletion_protection in ["enabled", "disabled"]:
            dp = deletion_protection
        else:
            raise ValueError("deletion_protection must be either 'enabled' or 'disabled'")

        fetched_tags = description.tags
        if fetched_tags is None:
            starting_tags: dict[str, str] = {}
        else:
            # Use getattr with a default to handle the 'object' type issue
            tags_obj = getattr(fetched_tags, "to_dict", None)
            if tags_obj is not None and callable(tags_obj):
                starting_tags = tags_obj()
            else:
                starting_tags = {}

        if tags is None:
            # Do not modify tags if none are provided
            tags = starting_tags
        else:
            # Merge existing tags with new tags
            tags = {**starting_tags, **tags}

        # Parse read_capacity if provided
        parsed_read_capacity = None
        if read_capacity is not None:
            parsed_read_capacity = PineconeDBControlRequestFactory.__parse_read_capacity(
                read_capacity
            )

        # Build deployment for pod configuration updates
        deployment_dict: dict[str, Any] | None = None
        if replicas is not None or pod_type is not None:
            pod_type_str = convert_enum_to_string(pod_type) if pod_type else None
            deployment_dict = {"deployment_type": "pod"}
            if replicas is not None:
                deployment_dict["replicas"] = replicas
            if pod_type_str is not None:
                deployment_dict["pod_type"] = pod_type_str

        # Note: embed configuration is no longer supported in alpha API configure_index
        # The schema field should be used instead for index configuration updates
        if embed is not None:
            raise NotImplementedError(
                "The 'embed' parameter is not supported in the alpha API. "
                "Use the 'schema' field for index configuration updates."
            )

        args_dict = parse_non_empty_args(
            [
                ("deletion_protection", dp),
                ("tags", IndexTags(**tags)),
                ("deployment", deployment_dict),
                ("read_capacity", parsed_read_capacity),
            ]
        )

        from typing import cast

        result = ConfigureIndexRequest(**args_dict)
        return cast(ConfigureIndexRequest, result)

    @staticmethod
    def create_collection_request(name: str, source: str) -> CreateCollectionRequest:
        from typing import cast

        result = CreateCollectionRequest(name=name, source=source)
        return cast(CreateCollectionRequest, result)
