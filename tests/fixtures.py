"""Test fixtures for creating test instances of SDK models.

This module provides helper functions for creating test instances with both
old-style and new-style API parameters for backward compatibility.
"""

from typing import Any
from pinecone.core.openapi.db_control.model.index_model import IndexModel as OpenAPIIndexModel
from pinecone.core.openapi.db_control.model.schema import Schema
from pinecone.core.openapi.db_control.model.schema_fields import SchemaFields
from pinecone.core.openapi.db_control.model.serverless_deployment import ServerlessDeployment
from pinecone.core.openapi.db_control.model.pod_deployment import PodDeployment
from pinecone.core.openapi.db_control.model.byoc_deployment import ByocDeployment
from pinecone.core.openapi.db_control.model.index_model_status import IndexModelStatus
from pinecone.db_control.models import IndexModel


def make_index_model(
    name: str = "test-index",
    dimension: int | None = None,
    metric: str | None = None,
    spec: dict[str, Any] | None = None,
    deployment: ServerlessDeployment | PodDeployment | ByocDeployment | None = None,
    schema: Schema | None = None,
    host: str = "https://test-index-1234.svc.pinecone.io",
    status: IndexModelStatus | None = None,
    **kwargs: Any,
) -> IndexModel:
    """Create an IndexModel for testing, accepting both old and new API parameters.

    This helper function allows tests to use either the old-style parameters
    (spec, dimension, metric) or new-style parameters (deployment, schema) for
    backward compatibility.

    Args:
        name: The index name
        dimension: Vector dimension (old-style, converted to schema)
        metric: Distance metric (old-style, converted to schema)
        spec: Index spec dict (old-style, converted to deployment)
        deployment: Deployment configuration (new-style)
        schema: Schema configuration (new-style)
        host: Index host URL
        status: Index status
        **kwargs: Additional fields to pass to IndexModel

    Returns:
        IndexModel wrapper instance

    Examples:
        Old-style API::

            index = make_index_model(
                name="my-index",
                dimension=1536,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )

        New-style API::

            index = make_index_model(
                name="my-index",
                schema=Schema(fields={"_values": SchemaFields(
                    type="dense_vector",
                    dimension=1536,
                    metric="cosine"
                )}),
                deployment=ServerlessDeployment(
                    deployment_type="serverless",
                    cloud="aws",
                    region="us-east-1"
                )
            )
    """
    # Convert old-style spec to deployment if needed
    if deployment is None and spec:
        deployment = _spec_to_deployment(spec)
    elif deployment is None:
        # Default serverless deployment
        deployment = ServerlessDeployment(
            deployment_type="serverless", cloud="aws", region="us-east-1"
        )

    # Convert dimension/metric to schema if needed
    if schema is None and (dimension or metric):
        schema = _create_schema_from_dimension_metric(dimension, metric)
    elif schema is None:
        # Default empty schema (for FTS-only indexes)
        schema = Schema(fields={})

    # Create status if needed
    if status is None:
        status = IndexModelStatus(ready=True, state="Ready")

    # Create the OpenAPI model
    openapi_model = OpenAPIIndexModel(
        name=name, schema=schema, deployment=deployment, host=host, status=status, **kwargs
    )

    # Wrap in the IndexModel wrapper
    return IndexModel(openapi_model)


def _spec_to_deployment(
    spec: dict[str, Any],
) -> ServerlessDeployment | PodDeployment | ByocDeployment:
    """Convert old-style spec dict to new-style Deployment.

    Args:
        spec: Spec dict with serverless, pod, or byoc key

    Returns:
        Appropriate Deployment instance
    """
    if "serverless" in spec:
        serverless_spec = spec["serverless"]
        return ServerlessDeployment(
            deployment_type="serverless",
            cloud=serverless_spec.get("cloud", "aws"),
            region=serverless_spec.get("region", "us-east-1"),
        )
    elif "pod" in spec:
        pod_spec = spec["pod"]
        return PodDeployment(
            deployment_type="pod",
            environment=pod_spec.get("environment", "us-east-1-aws"),
            pod_type=pod_spec.get("pod_type", "p1.x1"),
            replicas=pod_spec.get("replicas", 1),
            shards=pod_spec.get("shards", 1),
            pods=pod_spec.get("pods", 1),
        )
    elif "byoc" in spec:
        byoc_spec = spec["byoc"]
        return ByocDeployment(
            deployment_type="byoc", environment=byoc_spec.get("environment", "custom-env")
        )
    else:
        raise ValueError("spec must contain 'serverless', 'pod', or 'byoc' key")


def _create_schema_from_dimension_metric(dimension: int | None, metric: str | None) -> Schema:
    """Create a Schema from dimension and metric.

    Args:
        dimension: Vector dimension
        metric: Distance metric

    Returns:
        Schema with a dense_vector field
    """
    if dimension is None:
        # No dimension means FTS-only or sparse vectors
        # For simplicity, return empty schema
        return Schema(fields={})

    field_config: dict[str, Any] = {"type": "dense_vector", "dimension": dimension}

    if metric:
        field_config["metric"] = metric

    return Schema(fields={"_values": SchemaFields(**field_config)})
