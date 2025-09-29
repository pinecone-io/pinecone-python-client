import logging
from typing import Dict, Union

from pinecone.utils import parse_non_empty_args, convert_enum_to_string

from pinecone.core.openapi.repository_control.model.create_repository_request import (
    CreateRepositoryRequest as OpenAPICreateRepositoryRequest,
)
from pinecone.core.openapi.repository_control.model.repository_spec import (
    RepositorySpec as OpenAPIRepositorySpec,
)
from pinecone.core.openapi.repository_control.model.serverless_spec import (
    ServerlessSpec as OpenAPIServerlessSpec,
)
from pinecone.core.openapi.repository_control.model.document_schema import (
    DocumentSchema as OpenAPIDocumentSchema,
)
from pinecone.core.openapi.repository_control.model.document_schema_field_map import (
    DocumentSchemaFieldMap as OpenAPIDocumentSchemaFieldMap,
)
from pinecone.repository_control.models.serverless_spec import ServerlessSpec
from pinecone.repository_control.models.document_schema import DocumentSchema


logger = logging.getLogger(__name__)
""" :meta private: """


class PineconeRepositoryControlRequestFactory:
    """
    :meta private:

    This class facilitates translating user inputs into request objects.
    """

    @staticmethod
    def __parse_repository_spec(spec: Union[Dict, ServerlessSpec]) -> OpenAPIRepositorySpec:
        if isinstance(spec, dict):
            if "serverless" in spec:
                spec["serverless"]["cloud"] = convert_enum_to_string(spec["serverless"]["cloud"])
                spec["serverless"]["region"] = convert_enum_to_string(spec["serverless"]["region"])

                repository_spec = OpenAPIRepositorySpec(
                    serverless=OpenAPIServerlessSpec(**spec["serverless"])
                )
            else:
                raise ValueError("spec must contain a 'serverless' key")
        elif isinstance(spec, ServerlessSpec):
            repository_spec = OpenAPIRepositorySpec(
                serverless=OpenAPIServerlessSpec(cloud=spec.cloud, region=spec.region)
            )
        else:
            raise TypeError("spec must be of type dict or ServerlessSpec")

        return repository_spec

    @staticmethod
    def __parse_repository_schema(schema: Union[Dict, DocumentSchema]) -> OpenAPIDocumentSchema:
        if isinstance(schema, dict):
            if "fields" in schema:
                openapi_schema = OpenAPIDocumentSchema(
                    fields=OpenAPIDocumentSchemaFieldMap(**schema["fields"])
                )
                return openapi_schema
            else:
                raise ValueError("schema must contain a 'fields' key")
        elif isinstance(schema, DocumentSchema):
            # Extract the OpenAPI schema from the wrapper
            return schema.schema
        else:
            raise TypeError("schema must be of type dict or DocumentSchema")

    @staticmethod
    def create_repository_request(
        name: str, spec: Union[Dict, ServerlessSpec], schema: Union[Dict, DocumentSchema]
    ) -> OpenAPICreateRepositoryRequest:
        parsed_spec = PineconeRepositoryControlRequestFactory.__parse_repository_spec(spec)
        parsed_schema = PineconeRepositoryControlRequestFactory.__parse_repository_schema(schema)

        args = parse_non_empty_args([("name", name), ("spec", parsed_spec), ("schema", parsed_schema)])

        return OpenAPICreateRepositoryRequest(**args)
