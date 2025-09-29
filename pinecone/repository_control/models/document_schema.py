from pinecone.core.openapi.repository_control.model.document_schema import (
    DocumentSchema as OpenAPIDocumentSchema,
)
import json


class DocumentSchema:
    def __init__(self, schema: OpenAPIDocumentSchema):
        self.schema = schema

    def __str__(self):
        return str(self.schema)

    def __getattr__(self, attr):
        return getattr(self.schema, attr)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self):
        return self.schema.to_dict()
