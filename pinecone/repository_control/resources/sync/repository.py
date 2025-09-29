import time
import logging
from typing import Optional, Dict, Union, TYPE_CHECKING

from pinecone.repository_control.repository_host_store import RepositoryHostStore

from pinecone.repository_control.models import RepositoryModel, RepositoryList
from pinecone.utils import docslinks, require_kwargs, PluginAware

from pinecone.repository_control.request_factory import PineconeRepositoryControlRequestFactory
from pinecone.core.openapi.repository_control import API_VERSION

logger = logging.getLogger(__name__)
""" :meta private: """

if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration
    from pinecone.core.openapi.repository_control.api.manage_repositories_api import (
        ManageRepositoriesApi,
    )
    from pinecone.repository_control.models import ServerlessSpec, DocumentSchema


class RepositoryResource(PluginAware):
    def __init__(
        self,
        repository_api: "ManageRepositoriesApi",
        config: "Config",
        openapi_config: "OpenApiConfiguration",
        pool_threads: int,
    ):
        self._repository_api = repository_api
        """ :meta private: """

        self.config = config
        """ :meta private: """

        self._openapi_config = openapi_config
        """ :meta private: """

        self._pool_threads = pool_threads
        """ :meta private: """

        self._repository_host_store = RepositoryHostStore()
        """ :meta private: """

        super().__init__()  # Initialize PluginAware

    @require_kwargs
    def create(
        self,
        *,
        name: str,
        spec: Union[Dict, "ServerlessSpec"],
        schema: Union[Dict, "DocumentSchema"],
        timeout: Optional[int] = None,
    ) -> RepositoryModel:
        req = PineconeRepositoryControlRequestFactory.create_repository_request(
            name=name, spec=spec, schema=schema
        )
        resp = self._repository_api.create_repository(create_repository_request=req)

        if timeout == -1:
            return resp
        return self.__poll_describe_repository_until_ready(name, timeout)

    def __poll_describe_repository_until_ready(self, name: str, timeout: Optional[int] = None):
        total_wait_time = 0
        while True:
            description = self.describe(name=name)
            if description.status.state == "InitializationFailed":
                raise Exception(
                    f"Repository {name} failed to initialize. The repository status is {description.status.state}."
                )
            if description.status.ready:
                return description

            if timeout is not None and total_wait_time >= timeout:
                logger.error(
                    f"Repository {name} is not ready after {total_wait_time} seconds. Timeout reached."
                )
                link = docslinks["API_DESCRIBE_REPOSITORY"](API_VERSION)
                timeout_msg = f"Repository {name} is not ready after {total_wait_time} seconds. Please call describe_repository() to confirm repository status. See docs at {link}"
                raise TimeoutError(timeout_msg)

            logger.debug(
                f"Waiting for repository {name} to be ready. Total wait time {total_wait_time} seconds."
            )

            total_wait_time += 5
            time.sleep(5)

    @require_kwargs
    def delete(self, *, name: str, timeout: Optional[int] = None) -> None:
        self._repository_api.delete_repository(name)
        self._repository_host_store.delete_host(self.config, name)

        if timeout == -1:
            return

        if timeout is None:
            while self.has(name=name):
                time.sleep(5)
        else:
            while self.has(name=name) and timeout >= 0:
                time.sleep(5)
                timeout -= 5
        if timeout and timeout < 0:
            raise (
                TimeoutError(
                    "Please call the list_repositories API ({}) to confirm if repository is deleted".format(
                        "https://www.pinecone.io/docs/api/operation/list_repositories/"
                    )
                )
            )

    @require_kwargs
    def list(self) -> RepositoryList:
        response = self._repository_api.list_repositories()
        return RepositoryList(response)

    @require_kwargs
    def describe(self, *, name: str) -> RepositoryModel:
        api_instance = self._repository_api
        description = api_instance.describe_repository(name)
        host = description.host
        self._repository_host_store.set_host(self.config, name, host)

        return description

    @require_kwargs
    def has(self, *, name: str) -> bool:
        if name in self.list().names():
            return True
        else:
            return False

    def _get_host(self, name: str) -> str:
        """:meta private:"""
        return self._repository_host_store.get_host(
            api=self._repository_api, config=self.config, repository_name=name
        )
