import json
from pinecone.core.openapi.repository_control.model.repository_list import (
    RepositoryList as OpenAPIRepositoryList,
)
from .repository_model import RepositoryModel
from typing import List


class RepositoryList:
    def __init__(self, repository_list: OpenAPIRepositoryList):
        self.repository_list = repository_list
        self.repositories = [RepositoryModel(i) for i in self.repository_list.repositories]
        self.current = 0

    def names(self) -> List[str]:
        return [i.name for i in self.repositories]

    def __getitem__(self, key):
        return self.repositories[key]

    def __len__(self):
        return len(self.repositories)

    def __iter__(self):
        return iter(self.repositories)

    def __str__(self):
        return str(self.repositories)

    def __repr__(self):
        return json.dumps([i.to_dict() for i in self.repositories], indent=4)

    def __getattr__(self, attr):
        return getattr(self.repository_list, attr)
