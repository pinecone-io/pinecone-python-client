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
from pinecone.core.openapi.db_control.model.schema import Schema
from pinecone.core.openapi.db_control.model.schema_fields import SchemaFields
from pinecone.core.openapi.db_control.model.pod_deployment_metadata_config import (
    PodDeploymentMetadataConfig,
)
from pinecone.core.openapi.db_control.model.create_index_from_backup_request import (
    CreateIndexFromBackupRequest,
)
from pinecone.db_control.models import ServerlessSpec, PodSpec, ByocSpec, IndexModel, IndexEmbed
from pinecone.core.openapi.db_control.model.serverless_deployment import (
    ServerlessDeployment as ServerlessDeploymentModel,
)
from pinecone.core.openapi.db_control.model.pod_deployment import (
    PodDeployment as PodDeploymentModel,
)
from pinecone.core.openapi.db_control.model.byoc_deployment import (
    ByocDeployment as ByocDeploymentModel,
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
from .types import CreateIndexForModelEmbedTypedDict

if TYPE_CHECKING:
    from pinecone.db_control.models.serverless_spec import (
        ReadCapacityDict,
        MetadataSchemaFieldConfig,
    )
    from pinecone.core.openapi.db_control.model.read_capacity import ReadCapacity
    from pinecone.db_control.models.deployment import (
        ServerlessDeployment,
        PodDeployment,
        ByocDeployment,
    )

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
                # New API structure: ReadCapacityDedicatedSpec has node_type, scaling (object with strategy, replicas, shards)
                dedicated_dict: dict[str, Any] = read_capacity.get("dedicated", {})  # type: ignore

                if "node_type" not in dedicated_dict or dedicated_dict.get("node_type") is None:
                    raise ValueError(
                        "node_type is required when using Dedicated read capacity mode. "
                        "Please specify 'node_type' (e.g., 't1' or 'b1') in the 'dedicated' configuration."
                    )
                if "scaling" not in dedicated_dict or dedicated_dict.get("scaling") is None:
                    raise ValueError(
                        "scaling is required when using Dedicated read capacity mode. "
                        "Please specify 'scaling' (e.g., 'Manual') in the 'dedicated' configuration."
                    )

                node_type = dedicated_dict["node_type"]
                scaling_value = dedicated_dict["scaling"]

                # Handle manual scaling configuration
                from pinecone.core.openapi.db_control.model.read_capacity_dedicated_spec_response_scaling import (
                    ReadCapacityDedicatedSpecResponseScaling,
                )

                if isinstance(scaling_value, str):
                    # Old format: scaling is a string like "Manual", manual dict has shards/replicas
                    if scaling_value == "Manual":
                        if "manual" not in dedicated_dict or dedicated_dict.get("manual") is None:
                            raise ValueError(
                                "When using 'Manual' scaling with Dedicated read capacity mode, "
                                "the 'manual' field with 'shards' and 'replicas' is required."
                            )
                        manual_dict = dedicated_dict["manual"]
                        if not isinstance(manual_dict, dict):
                            raise ValueError(
                                "The 'manual' field must be a dictionary with 'shards' and 'replicas' keys."
                            )
                        if "shards" not in manual_dict or "replicas" not in manual_dict:
                            missing = []
                            if "shards" not in manual_dict:
                                missing.append("shards")
                            if "replicas" not in manual_dict:
                                missing.append("replicas")
                            raise ValueError(
                                f"The 'manual' configuration is missing required fields: {', '.join(missing)}."
                            )

                        # Create scaling object with strategy, replicas, shards
                        scaling_obj = ReadCapacityDedicatedSpecResponseScaling(
                            strategy=scaling_value,
                            replicas=manual_dict["replicas"],
                            shards=manual_dict["shards"],
                        )
                    else:
                        # Other scaling strategies might not need manual config
                        scaling_obj = ReadCapacityDedicatedSpecResponseScaling(
                            strategy=scaling_value, replicas=1, shards=1
                        )
                else:
                    # Already a scaling object
                    scaling_obj = scaling_value

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
    def __parse_schema(
        schema: (
            dict[
                str, "MetadataSchemaFieldConfig"
            ]  # Direct field mapping: {field_name: {filterable: bool}}
            | dict[
                str, dict[str, Any]
            ]  # Dict with "fields" wrapper: {"fields": {field_name: {...}}, ...}
            | Schema  # OpenAPI model instance
        ),
    ) -> Schema:
        """Parse schema dict into Schema instance.

        :param schema: Dict with schema configuration (either {field_name: {filterable: bool, ...}} or
            {"fields": {field_name: {filterable: bool, ...}}, ...}) or existing Schema instance
        :return: Schema instance
        """
        if isinstance(schema, dict):
            schema_kwargs: dict[str, Any] = {}
            # Handle two formats:
            # 1. {field_name: {filterable: bool, ...}} - direct field mapping
            # 2. {"fields": {field_name: {filterable: bool, ...}}, ...} - with fields wrapper
            if "fields" in schema:
                # Format 2: has fields wrapper
                fields = {}
                for field_name, field_config in schema["fields"].items():
                    if isinstance(field_config, dict):
                        # Pass through the entire field_config dict to allow future API fields
                        fields[field_name] = SchemaFields(**field_config)
                    else:
                        # If not a dict, create with default filterable=True
                        fields[field_name] = SchemaFields(filterable=True)
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
                        # Pass through the entire field_config dict to allow future API fields
                        fields[field_name] = SchemaFields(**field_config)
                    else:
                        # If not a dict, create with default filterable=True
                        fields[field_name] = SchemaFields(filterable=True)
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

            result = Schema(**schema_kwargs)
            return cast(Schema, result)
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
        deployment: ServerlessDeploymentModel | PodDeploymentModel | ByocDeploymentModel
        if isinstance(spec, dict):
            if "serverless" in spec:
                serverless_spec = spec["serverless"]
                # Convert enum values to strings for consistency with __parse_index_spec
                cloud = convert_enum_to_string(serverless_spec.get("cloud", ""))
                region = convert_enum_to_string(serverless_spec.get("region", ""))
                deployment = ServerlessDeploymentModel(
                    deployment_type="serverless", cloud=cloud, region=region
                )
                deployment_dict = deployment.to_dict()
            elif "pod" in spec:
                pod_spec = spec["pod"]
                # Convert enum values to strings for consistency with __parse_index_spec
                environment = convert_enum_to_string(pod_spec.get("environment", ""))
                pod_type = convert_enum_to_string(pod_spec.get("pod_type", "p1.x1"))
                pod_kwargs: dict[str, Any] = {
                    "deployment_type": "pod",
                    "environment": environment,
                    "pod_type": pod_type,
                    "replicas": pod_spec.get("replicas") or 1,  # Use default 1 if None or missing
                    "shards": pod_spec.get("shards") or 1,  # Use default 1 if None or missing
                }
                # Only include pods if it's explicitly provided and not None
                pods_value = pod_spec.get("pods")
                if pods_value is not None:
                    pod_kwargs["pods"] = pods_value
                # Handle metadata_config if present - convert to PodDeploymentMetadataConfig
                if pod_spec.get("metadata_config"):
                    metadata_config_dict = pod_spec["metadata_config"]
                    if isinstance(metadata_config_dict, dict) and metadata_config_dict:
                        pod_kwargs["metadata_config"] = PodDeploymentMetadataConfig(
                            **metadata_config_dict
                        )
                deployment = PodDeploymentModel(**pod_kwargs)
                deployment_dict = deployment.to_dict()
            elif "byoc" in spec:
                byoc_spec = spec["byoc"]
                # Convert enum values to strings for consistency with __parse_index_spec
                environment = convert_enum_to_string(byoc_spec.get("environment", ""))
                deployment = ByocDeploymentModel(deployment_type="byoc", environment=environment)
                deployment_dict = deployment.to_dict()
            else:
                raise ValueError("spec must contain either 'serverless', 'pod', or 'byoc' key")
        elif isinstance(spec, ServerlessSpec):
            deployment = ServerlessDeploymentModel(
                deployment_type="serverless", cloud=spec.cloud, region=spec.region
            )
            deployment_dict = deployment.to_dict()
        elif isinstance(spec, PodSpec):
            # PodDeployment requires pod_type, but PodSpec defaults to "p1.x1"
            pod_type = spec.pod_type if spec.pod_type is not None else "p1.x1"
            # Use explicit None check to preserve 0 values (consistent with dict handling)
            replicas = spec.replicas if spec.replicas is not None else 1
            shards = spec.shards if spec.shards is not None else 1
            pod_obj_kwargs: dict[str, Any] = {
                "deployment_type": "pod",
                "environment": spec.environment,
                "pod_type": pod_type,
                "replicas": replicas,
                "shards": shards,
            }
            # Only include pods if it's not None
            if spec.pods is not None:
                pod_obj_kwargs["pods"] = spec.pods
            # Handle metadata_config if present - convert to PodDeploymentMetadataConfig
            if spec.metadata_config:
                pod_obj_kwargs["metadata_config"] = PodDeploymentMetadataConfig(
                    **spec.metadata_config
                )
            deployment = PodDeploymentModel(**pod_obj_kwargs)
            deployment_dict = deployment.to_dict()
        elif isinstance(spec, ByocSpec):
            deployment = ByocDeploymentModel(deployment_type="byoc", environment=spec.environment)
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
        deployment = ServerlessDeploymentModel(
            deployment_type="serverless", cloud=cloud, region=region
        )
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
        deployment: "ServerlessDeployment | PodDeployment | ByocDeployment | None" = None,
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
            from pinecone.db_control.models.deployment import (
                ServerlessDeployment as SDKServerlessDeployment,
            )

            deployment = SDKServerlessDeployment(cloud="aws", region="us-east-1")

        # Convert SDK deployment to dict, then let CreateIndexRequest deserialize it
        deployment_dict = deployment.to_dict()

        # Import the Deployment union class and use it to properly deserialize
        from pinecone.core.openapi.db_control.model.deployment import Deployment

        parsed_deployment = Deployment(**deployment_dict)

        # Parse schema
        schema_dict = PineconeDBControlRequestFactory._serialize_schema(schema)
        parsed_schema = None
        if schema_dict and schema_dict.get("fields"):
            parsed_schema = PineconeDBControlRequestFactory.__parse_schema(schema_dict)

        args = parse_non_empty_args(
            [
                ("name", name),
                ("deployment", parsed_deployment),
                ("schema", parsed_schema),
                ("deletion_protection", dp),
                ("tags", tags_obj),
            ]
        )

        from typing import cast

        result = CreateIndexRequest(**args)
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
        """Create a CreateIndexRequest using legacy parameters.

        This method translates legacy spec/dimension/metric parameters to the new
        deployment/schema format required by the current API.
        """
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

        # Translate legacy parameters to new deployment + schema format
        deployment_dict, schema_dict = PineconeDBControlRequestFactory._translate_legacy_request(
            spec=spec, dimension=dimension, metric=metric, vector_type=vector_type
        )

        # Convert metadata_config dict to PodDeploymentMetadataConfig if present
        if deployment_dict.get("deployment_type") == "pod" and "metadata_config" in deployment_dict:
            metadata_config_dict = deployment_dict["metadata_config"]
            if isinstance(metadata_config_dict, dict):
                deployment_dict["metadata_config"] = PodDeploymentMetadataConfig(
                    **metadata_config_dict
                )

        # Convert deployment_dict to Deployment object using the discriminator
        from pinecone.core.openapi.db_control.model.deployment import Deployment

        parsed_deployment = Deployment(**deployment_dict)

        # Convert schema_dict to Schema object if present
        parsed_schema = None
        if schema_dict and schema_dict.get("fields"):
            parsed_schema = PineconeDBControlRequestFactory.__parse_schema(schema_dict)

        args = parse_non_empty_args(
            [
                ("name", name),
                ("deployment", parsed_deployment),
                ("schema", parsed_schema),
                ("deletion_protection", dp),
                ("tags", tags_obj),
            ]
        )

        from typing import cast

        result = CreateIndexRequest(**args)
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
            | Schema  # OpenAPI model instance
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
        read_capacity: (
            "ReadCapacityDict"
            | "ReadCapacity"
            | ReadCapacityOnDemandSpec
            | ReadCapacityDedicatedSpec
        )
        | None = None,
    ):
        if deletion_protection is None:
            dp = description.deletion_protection
        elif isinstance(deletion_protection, DeletionProtection):
            dp = deletion_protection.value
        elif deletion_protection in ["enabled", "disabled"]:
            dp = deletion_protection
        else:
            raise ValueError("deletion_protection must be either 'enabled' or 'disabled'")

        fetched_tags = description.tags
        starting_tags: dict[str, str]
        if fetched_tags is None:
            starting_tags = {}
        else:
            starting_tags = fetched_tags.to_dict() if hasattr(fetched_tags, "to_dict") else {}

        if tags is None:
            # Do not modify tags if none are provided
            tags = starting_tags
        else:
            # Merge existing tags with new tags
            tags = {**starting_tags, **tags}

        pod_config_args: dict[str, Any] = {}
        if pod_type:
            new_pod_type = convert_enum_to_string(pod_type)
            pod_config_args.update(pod_type=new_pod_type)
        if replicas:
            pod_config_args.update(replicas=replicas)

        # Parse read_capacity if provided
        parsed_read_capacity = None
        if read_capacity is not None:
            parsed_read_capacity = PineconeDBControlRequestFactory.__parse_read_capacity(
                read_capacity
            )

        spec = None
        if pod_config_args:
            spec = {"pod": pod_config_args}
        elif parsed_read_capacity is not None:
            # Serverless index configuration
            spec = {"serverless": {"read_capacity": parsed_read_capacity}}

        args_dict = parse_non_empty_args(
            [("deletion_protection", dp), ("tags", IndexTags(**tags)), ("spec", spec)]
        )

        from typing import cast

        result = ConfigureIndexRequest(**args_dict)
        return cast(ConfigureIndexRequest, result)

    @staticmethod
    def create_collection_request(name: str, source: str) -> CreateCollectionRequest:
        from typing import cast

        result = CreateCollectionRequest(name=name, source=source)
        return cast(CreateCollectionRequest, result)
