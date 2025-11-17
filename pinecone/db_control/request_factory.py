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
from pinecone.core.openapi.db_control.model.configure_index_request_embed import (
    ConfigureIndexRequestEmbed,
)
from pinecone.core.openapi.db_control.model.index_spec import IndexSpec
from pinecone.core.openapi.db_control.model.index_tags import IndexTags
from pinecone.core.openapi.db_control.model.serverless_spec import (
    ServerlessSpec as ServerlessSpecModel,
)
from pinecone.core.openapi.db_control.model.read_capacity_on_demand_spec import (
    ReadCapacityOnDemandSpec,
)
from pinecone.core.openapi.db_control.model.read_capacity_dedicated_spec import (
    ReadCapacityDedicatedSpec,
)
from pinecone.core.openapi.db_control.model.read_capacity_dedicated_config import (
    ReadCapacityDedicatedConfig,
)
from pinecone.core.openapi.db_control.model.scaling_config_manual import ScalingConfigManual
from pinecone.core.openapi.db_control.model.backup_model_schema import BackupModelSchema
from pinecone.core.openapi.db_control.model.backup_model_schema_fields import (
    BackupModelSchemaFields,
)
from pinecone.core.openapi.db_control.model.byoc_spec import ByocSpec as ByocSpecModel
from pinecone.core.openapi.db_control.model.pod_spec import PodSpec as PodSpecModel
from pinecone.core.openapi.db_control.model.pod_spec_metadata_config import PodSpecMetadataConfig
from pinecone.core.openapi.db_control.model.create_index_from_backup_request import (
    CreateIndexFromBackupRequest,
)
from pinecone.db_control.models import ServerlessSpec, PodSpec, ByocSpec, IndexModel, IndexEmbed

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
                # Construct ReadCapacityDedicatedConfig
                # node_type and scaling are required fields
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
                scaling = dedicated_dict["scaling"]
                dedicated_config_kwargs = {"node_type": node_type, "scaling": scaling}

                # Validate that manual scaling configuration is provided when scaling is "Manual"
                if scaling == "Manual":
                    if "manual" not in dedicated_dict or dedicated_dict.get("manual") is None:
                        raise ValueError(
                            "When using 'Manual' scaling with Dedicated read capacity mode, "
                            "the 'manual' field with 'shards' and 'replicas' is required. "
                            "Please specify 'manual': {'shards': <number>, 'replicas': <number>} "
                            "in the 'dedicated' configuration."
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
                            f"The 'manual' configuration is missing required fields: {', '.join(missing)}. "
                            "Please provide both 'shards' and 'replicas' in the 'manual' configuration."
                        )
                    dedicated_config_kwargs["manual"] = ScalingConfigManual(**manual_dict)
                elif "manual" in dedicated_dict:
                    # Allow manual to be provided for other scaling types (future compatibility)
                    manual_dict = dedicated_dict["manual"]
                    dedicated_config_kwargs["manual"] = ScalingConfigManual(**manual_dict)

                dedicated_config = ReadCapacityDedicatedConfig(**dedicated_config_kwargs)
                result = ReadCapacityDedicatedSpec(mode="Dedicated", dedicated=dedicated_config)
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
            | BackupModelSchema  # OpenAPI model instance
        ),
    ) -> BackupModelSchema:
        """Parse schema dict into BackupModelSchema instance.

        :param schema: Dict with schema configuration (either {field_name: {filterable: bool, ...}} or
            {"fields": {field_name: {filterable: bool, ...}}, ...}) or existing BackupModelSchema instance
        :return: BackupModelSchema instance
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
                        fields[field_name] = BackupModelSchemaFields(**field_config)
                    else:
                        # If not a dict, create with default filterable=True
                        fields[field_name] = BackupModelSchemaFields(filterable=True)
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
                        fields[field_name] = BackupModelSchemaFields(**field_config)
                    else:
                        # If not a dict, create with default filterable=True
                        fields[field_name] = BackupModelSchemaFields(filterable=True)
                # Ensure fields is always set, even if empty
                schema_kwargs["fields"] = fields

            # Validate that fields is present before constructing BackupModelSchema
            if "fields" not in schema_kwargs:
                raise ValueError(
                    "Schema dict must contain field definitions. "
                    "Either provide a 'fields' key with field configurations, "
                    "or provide field_name: field_config pairs directly."
                )

            from typing import cast

            result = BackupModelSchema(**schema_kwargs)
            return cast(BackupModelSchema, result)
        else:
            # Already a BackupModelSchema instance
            return schema

    @staticmethod
    def __parse_index_spec(spec: Dict | ServerlessSpec | PodSpec | ByocSpec) -> IndexSpec:
        if isinstance(spec, dict):
            if "serverless" in spec:
                spec["serverless"]["cloud"] = convert_enum_to_string(spec["serverless"]["cloud"])
                spec["serverless"]["region"] = convert_enum_to_string(spec["serverless"]["region"])

                # Handle read_capacity if present
                if "read_capacity" in spec["serverless"]:
                    spec["serverless"]["read_capacity"] = (
                        PineconeDBControlRequestFactory.__parse_read_capacity(
                            spec["serverless"]["read_capacity"]
                        )
                    )

                # Handle schema if present - convert to BackupModelSchema
                if "schema" in spec["serverless"]:
                    schema_dict = spec["serverless"]["schema"]
                    if isinstance(schema_dict, dict):
                        # Process fields if present, otherwise pass through as-is
                        schema_kwargs = {}
                        if "fields" in schema_dict:
                            fields = {}
                            for field_name, field_config in schema_dict["fields"].items():
                                if isinstance(field_config, dict):
                                    # Pass through the entire field_config dict to allow future API fields
                                    fields[field_name] = BackupModelSchemaFields(**field_config)
                                else:
                                    # If not a dict, create with default filterable=True
                                    fields[field_name] = BackupModelSchemaFields(filterable=True)
                            schema_kwargs["fields"] = fields

                        # Pass through any other fields in schema_dict to allow future API fields
                        for key, value in schema_dict.items():
                            if key != "fields":
                                schema_kwargs[key] = value

                        spec["serverless"]["schema"] = BackupModelSchema(**schema_kwargs)

                index_spec = IndexSpec(serverless=ServerlessSpecModel(**spec["serverless"]))
            elif "pod" in spec:
                spec["pod"]["environment"] = convert_enum_to_string(spec["pod"]["environment"])
                args_dict = parse_non_empty_args(
                    [
                        ("environment", spec["pod"].get("environment")),
                        ("metadata_config", spec["pod"].get("metadata_config")),
                        ("replicas", spec["pod"].get("replicas")),
                        ("shards", spec["pod"].get("shards")),
                        ("pods", spec["pod"].get("pods")),
                        ("source_collection", spec["pod"].get("source_collection")),
                    ]
                )
                if args_dict.get("metadata_config"):
                    args_dict["metadata_config"] = PodSpecMetadataConfig(
                        indexed=args_dict["metadata_config"].get("indexed", None)
                    )
                index_spec = IndexSpec(pod=PodSpecModel(**args_dict))
            elif "byoc" in spec:
                index_spec = IndexSpec(byoc=ByocSpecModel(**spec["byoc"]))
            else:
                raise ValueError("spec must contain either 'serverless', 'pod', or 'byoc' key")
        elif isinstance(spec, ServerlessSpec):
            # Build args dict for ServerlessSpecModel
            serverless_args: dict[str, Any] = {"cloud": spec.cloud, "region": spec.region}

            # Handle read_capacity
            if spec.read_capacity is not None:
                serverless_args["read_capacity"] = (
                    PineconeDBControlRequestFactory.__parse_read_capacity(spec.read_capacity)
                )

            # Handle schema
            if spec.schema is not None:
                # Convert dict to BackupModelSchema
                # schema is {field_name: {filterable: bool, ...}}
                # Pass through the entire field_config to allow future API fields
                fields = {}
                for field_name, field_config in spec.schema.items():
                    if isinstance(field_config, dict):
                        # Pass through the entire field_config dict to allow future API fields
                        fields[field_name] = BackupModelSchemaFields(**field_config)
                    else:
                        # If not a dict, create with default filterable=True
                        fields[field_name] = BackupModelSchemaFields(filterable=True)
                serverless_args["schema"] = BackupModelSchema(fields=fields)

            index_spec = IndexSpec(serverless=ServerlessSpecModel(**serverless_args))
        elif isinstance(spec, PodSpec):
            args_dict = parse_non_empty_args(
                [
                    ("replicas", spec.replicas),
                    ("shards", spec.shards),
                    ("pods", spec.pods),
                    ("source_collection", spec.source_collection),
                ]
            )
            if spec.metadata_config:
                args_dict["metadata_config"] = PodSpecMetadataConfig(
                    indexed=spec.metadata_config.get("indexed", None)
                )

            index_spec = IndexSpec(
                pod=PodSpecModel(environment=spec.environment, pod_type=spec.pod_type, **args_dict)
            )
        elif isinstance(spec, ByocSpec):
            args_dict = parse_non_empty_args([("environment", spec.environment)])
            index_spec = IndexSpec(byoc=ByocSpecModel(**args_dict))
        else:
            raise TypeError("spec must be of type dict, ServerlessSpec, PodSpec, or ByocSpec")

        from typing import cast

        return cast(IndexSpec, index_spec)

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
        index_spec = PineconeDBControlRequestFactory.__parse_index_spec(spec)

        if vector_type == VectorType.SPARSE.value and dimension is not None:
            raise ValueError("dimension should not be specified for sparse indexes")

        args = parse_non_empty_args(
            [
                ("name", name),
                ("dimension", dimension),
                ("metric", metric),
                ("spec", index_spec),
                ("deletion_protection", dp),
                ("vector_type", vector_type),
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
            | BackupModelSchema  # OpenAPI model instance
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
        if fetched_tags is None:
            starting_tags = {}
        else:
            starting_tags = fetched_tags.to_dict()

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

        embed_config = None
        if embed is not None:
            embed_config = ConfigureIndexRequestEmbed(**dict(embed))

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
            [
                ("deletion_protection", dp),
                ("tags", IndexTags(**tags)),
                ("spec", spec),
                ("embed", embed_config),
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
