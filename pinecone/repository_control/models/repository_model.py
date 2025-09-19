from pinecone.core.openapi.repository_control.model.repository_model import (
    RepositoryModel as OpenAPIRepositoryModel,
)
import json


class RepositoryModel:
    def __init__(self, repository: OpenAPIRepositoryModel):
        self.repository = repository

    def __str__(self):
        return str(self.repository)

    def __getattr__(self, attr):
        return getattr(self.repository, attr)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self):
        return self.repository.to_dict()
