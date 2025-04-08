import time
import logging
from typing import Optional, Dict, Union

from ..index_host_store import IndexHostStore

from pinecone.models import ServerlessSpec, PodSpec, IndexModel, IndexList, IndexEmbed
from pinecone.utils import docslinks

from pinecone.enums import (
    Metric,
    VectorType,
    DeletionProtection,
    PodType,
    CloudProvider,
    AwsRegion,
    GcpRegion,
    AzureRegion,
)
from ..types import CreateIndexForModelEmbedTypedDict
from ..request_factory import PineconeDBControlRequestFactory

logger = logging.getLogger(__name__)
""" @private """


class IndexResource:
    def __init__(self, index_api, config):
        self.index_api = index_api
        """ @private """

        self.config = config
        """ @private """

        self.index_host_store = IndexHostStore()
        """ @private """

    def create(
        self,
        name: str,
        spec: Union[Dict, ServerlessSpec, PodSpec],
        dimension: Optional[int] = None,
        metric: Optional[Union[Metric, str]] = Metric.COSINE,
        timeout: Optional[int] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = DeletionProtection.DISABLED,
        vector_type: Optional[Union[VectorType, str]] = VectorType.DENSE,
        tags: Optional[Dict[str, str]] = None,
    ) -> IndexModel:
        req = PineconeDBControlRequestFactory.create_index_request(
            name=name,
            spec=spec,
            dimension=dimension,
            metric=metric,
            deletion_protection=deletion_protection,
            vector_type=vector_type,
            tags=tags,
        )
        resp = self.index_api.create_index(create_index_request=req)

        if timeout == -1:
            return IndexModel(resp)
        return self.__poll_describe_index_until_ready(name, timeout)

    def create_for_model(
        self,
        name: str,
        cloud: Union[CloudProvider, str],
        region: Union[AwsRegion, GcpRegion, AzureRegion, str],
        embed: Union[IndexEmbed, CreateIndexForModelEmbedTypedDict],
        tags: Optional[Dict[str, str]] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = DeletionProtection.DISABLED,
        timeout: Optional[int] = None,
    ) -> IndexModel:
        req = PineconeDBControlRequestFactory.create_index_for_model_request(
            name=name,
            cloud=cloud,
            region=region,
            embed=embed,
            tags=tags,
            deletion_protection=deletion_protection,
        )
        resp = self.index_api.create_index_for_model(req)

        if timeout == -1:
            return IndexModel(resp)
        return self.__poll_describe_index_until_ready(name, timeout)

    def __poll_describe_index_until_ready(self, name: str, timeout: Optional[int] = None):
        description = None

        def is_ready() -> bool:
            nonlocal description
            description = self.describe(name=name)
            return description.status.ready

        total_wait_time = 0
        if timeout is None:
            # Wait indefinitely
            while not is_ready():
                logger.debug(
                    f"Waiting for index {name} to be ready. Total wait time {total_wait_time} seconds."
                )
                total_wait_time += 5
                time.sleep(5)

        else:
            # Wait for a maximum of timeout seconds
            while not is_ready():
                if timeout < 0:
                    logger.error(f"Index {name} is not ready. Timeout reached.")
                    link = docslinks["API_DESCRIBE_INDEX"]
                    timeout_msg = (
                        f"Please call describe_index() to confirm index status. See docs at {link}"
                    )
                    raise TimeoutError(timeout_msg)

                logger.debug(
                    f"Waiting for index {name} to be ready. Total wait time: {total_wait_time}"
                )
                total_wait_time += 5
                time.sleep(5)
                timeout -= 5

        return description

    def delete(self, name: str, timeout: Optional[int] = None):
        self.index_api.delete_index(name)
        self.index_host_store.delete_host(self.config, name)

        if timeout == -1:
            return

        if timeout is None:
            while self.has_index(name):
                time.sleep(5)
        else:
            while self.has_index(name) and timeout >= 0:
                time.sleep(5)
                timeout -= 5
        if timeout and timeout < 0:
            raise (
                TimeoutError(
                    "Please call the list_indexes API ({}) to confirm if index is deleted".format(
                        "https://www.pinecone.io/docs/api/operation/list_indexes/"
                    )
                )
            )

    def list(self) -> IndexList:
        response = self.index_api.list_indexes()
        return IndexList(response)

    def describe(self, name: str) -> IndexModel:
        api_instance = self.index_api
        description = api_instance.describe_index(name)
        host = description.host
        self.index_host_store.set_host(self.config, name, host)

        return IndexModel(description)

    def has(self, name: str) -> bool:
        if name in self.list().names():
            return True
        else:
            return False

    def configure(
        self,
        name: str,
        replicas: Optional[int] = None,
        pod_type: Optional[Union[PodType, str]] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        api_instance = self.index_api
        description = self.describe(name=name)

        req = PineconeDBControlRequestFactory.configure_index_request(
            description=description,
            replicas=replicas,
            pod_type=pod_type,
            deletion_protection=deletion_protection,
            tags=tags,
        )
        api_instance.configure_index(name, configure_index_request=req)
