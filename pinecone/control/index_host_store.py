from typing import Dict
from pinecone.config import Config
from pinecone.core.client.api.manage_pod_indexes_api import ManagePodIndexesApi as IndexOperationsApi

class SingletonMeta(type):
    _instances: Dict[str, str] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class IndexHostStore(metaclass=SingletonMeta):
    def __init__(self):
        self._indexHosts = {}

    def _key(self, config: Config, index_name: str) -> str:
        return ":".join([config.api_key, index_name])

    def delete_host(self, config: Config, index_name: str):
        key = self._key(config, index_name)
        if key in self._indexHosts:
            del self._indexHosts[key]

    def set_host(self, config: Config, index_name: str, host: str):
        if host:
            key = self._key(config, index_name)
            self._indexHosts[key] = "https://" + host

    def get_host(self, api: IndexOperationsApi, config: Config, index_name: str) -> str:
        key = self._key(config, index_name)
        if key in self._indexHosts:
            return self._indexHosts[key]
        else:
            description = api.describe_index(index_name)
            self.set_host(config, index_name, description.status.host)
            return self._indexHosts[key]
