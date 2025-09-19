import logging
from typing import Dict, Union

from pinecone.utils import parse_non_empty_args, convert_enum_to_string

from pinecone.core.openapi.repository_control.model.create_repository_request import (
    CreateRepositoryRequest,
)
from pinecone.core.openapi.repository_control.model.repository_spec import RepositorySpec
from pinecone.core.openapi.repository_control.model.serverless_spec import (
    ServerlessSpec as ServerlessSpecModel,
)
from pinecone.core.openapi.repository_control.model.serverless_spec import ServerlessSpec
from pinecone.core.openapi.repository_control.model.document_schema import DocumentSchema
from pinecone.core.openapi.repository_control.model.document_schema_field_map import (
    DocumentSchemaFieldMap,
)


logger = logging.getLogger(__name__)
""" :meta private: """


class PineconeRepositoryControlRequestFactory:
    """
    :meta private:

    This class facilitates translating user inputs into request objects.
    """

    @staticmethod
    def __parse_repository_spec(spec: Union[Dict, ServerlessSpec]) -> RepositorySpec:
        if isinstance(spec, dict):
            if "serverless" in spec:
                spec["serverless"]["cloud"] = convert_enum_to_string(spec["serverless"]["cloud"])
                spec["serverless"]["region"] = convert_enum_to_string(spec["serverless"]["region"])

                repository_spec = RepositorySpec(
                    serverless=ServerlessSpecModel(**spec["serverless"])
                )
            else:
                raise ValueError("spec must contain a 'serverless' key")
        elif isinstance(spec, ServerlessSpec):
            repository_spec = RepositorySpec(
                serverless=ServerlessSpecModel(cloud=spec.cloud, region=spec.region)
            )
        else:
            raise TypeError("spec must be of type dict or ServerlessSpec")

        return repository_spec

    @staticmethod
    def __parse_repository_schema(schema: Union[Dict, DocumentSchema]) -> DocumentSchema:
        if isinstance(schema, dict):
            if "fields" in schema:
                repository_schema = DocumentSchema(
                    fields=DocumentSchemaFieldMap(**schema["fields"])
                )
            else:
                raise ValueError("schema must contain a 'fields' key")
        elif isinstance(schema, DocumentSchema):
            repository_schema = schema
        else:
            raise TypeError("schema must be of type dict or DocumentSchema")

        return repository_schema

    @staticmethod
    def create_repository_request(
        name: str, spec: Union[Dict, ServerlessSpec], schema: Union[Dict, DocumentSchema]
    ) -> CreateRepositoryRequest:
        spec = PineconeRepositoryControlRequestFactory.__parse_repository_spec(spec)
        schema = PineconeRepositoryControlRequestFactory.__parse_repository_schema(schema)

        args = parse_non_empty_args([("name", name), ("spec", spec), ("schema", schema)])

        return CreateRepositoryRequest(**args)
