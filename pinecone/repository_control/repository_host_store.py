from typing import Dict
from pinecone.config import Config
from pinecone.core.openapi.repository_control.api.manage_repositories_api import (
    ManageRepositoriesApi as RepositoriesOperationsApi,
)
from pinecone.openapi_support.exceptions import PineconeException
from pinecone.utils import normalize_host


class SingletonMeta(type):
    _instances: Dict[str, str] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class RepositoryHostStore(metaclass=SingletonMeta):
    _repositoryHosts: Dict[str, str]

    def __init__(self) -> None:
        self._repositoryHosts = {}

    def _key(self, config: Config, repository_name: str) -> str:
        return ":".join([config.api_key, repository_name])

    def delete_host(self, config: Config, repository_name: str):
        key = self._key(config, repository_name)
        if key in self._repositoryHosts:
            del self._repositoryHosts[key]

    def key_exists(self, key: str) -> bool:
        return key in self._repositoryHosts

    def set_host(self, config: Config, repository_name: str, host: str):
        if host:
            key = self._key(config, repository_name)
            self._repositoryHosts[key] = normalize_host(host)

    def get_host(self, api: RepositoriesOperationsApi, config: Config, repository_name: str) -> str:
        key = self._key(config, repository_name)
        if self.key_exists(key):
            return self._repositoryHosts[key]
        else:
            description = api.describe_repository(repository_name)
            self.set_host(config, repository_name, description.host)
            if not self.key_exists(key):
                raise PineconeException(
                    f"Could not get host for repository: {repository_name}. Call describe_repository('{repository_name}') to check the current status."
                )
            return self._repositoryHosts[key]
