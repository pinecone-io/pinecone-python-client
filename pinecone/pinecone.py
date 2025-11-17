from __future__ import annotations

import logging
from typing import Dict, TYPE_CHECKING, Any, NoReturn
from multiprocessing import cpu_count
import warnings

from pinecone.config import PineconeConfig, ConfigBuilder

from pinecone.utils import normalize_host, PluginAware, docslinks, require_kwargs
from .langchain_import_warnings import _build_langchain_attribute_error_message

logger = logging.getLogger(__name__)
""" :meta private: """

if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration
    from pinecone.db_data import _Index as Index, _IndexAsyncio as IndexAsyncio
    from pinecone.db_control.index_host_store import IndexHostStore
    from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi
    from pinecone.inference import Inference
    from pinecone.db_control import DBControl
    from pinecone.db_control.types import CreateIndexForModelEmbedTypedDict, ConfigureIndexEmbed
    from pinecone.db_control.models.serverless_spec import (
        ReadCapacityDict,
        MetadataSchemaFieldConfig,
    )
    from pinecone.core.openapi.db_control.model.read_capacity import ReadCapacity
    from pinecone.core.openapi.db_control.model.read_capacity_on_demand_spec import (
        ReadCapacityOnDemandSpec,
    )
    from pinecone.core.openapi.db_control.model.read_capacity_dedicated_spec import (
        ReadCapacityDedicatedSpec,
    )
    from pinecone.core.openapi.db_control.model.backup_model_schema import BackupModelSchema
    from pinecone.db_control.enums import (
        Metric,
        VectorType,
        DeletionProtection,
        PodType,
        CloudProvider,
        AwsRegion,
        GcpRegion,
        AzureRegion,
    )
    from pinecone.db_control.models import (
        ServerlessSpec,
        PodSpec,
        ByocSpec,
        IndexModel,
        IndexList,
        CollectionList,
        IndexEmbed,
        BackupModel,
        BackupList,
        RestoreJobModel,
        RestoreJobList,
    )


class Pinecone(PluginAware):
    """
    A client for interacting with Pinecone APIs.
    """

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        proxy_url: str | None = None,
        proxy_headers: dict[str, str] | None = None,
        ssl_ca_certs: str | None = None,
        ssl_verify: bool | None = None,
        additional_headers: dict[str, str] | None = {},
        pool_threads: int | None = None,
        **kwargs,
    ) -> None:
        """
        The ``Pinecone`` class is the main entry point for interacting with Pinecone via this Python SDK.
        Instances of the ``Pinecone`` class are used to manage and interact with Pinecone resources such as
        indexes, backups, and collections. When using the SDK, calls are made on your behalf to the API
        documented at `https://docs.pinecone.io <https://docs.pinecone.io/reference/api/introduction>`_.

        The class also holds inference functionality (embed, rerank) under the ``inference`` namespace.


        When you are ready to perform data operations on an index, you will need to instantiate an index client. Though the functionality of the index client is defined in a different
        class, it is instantiated through the ``Index()`` method in order for configurations to be shared between the two objects.

        :param api_key: The API key to use for authentication. If not passed via kwarg, the API key will be read from the environment variable ``PINECONE_API_KEY``.
        :type api_key: str, optional
        :param host: The control plane host. If unspecified, the host ``api.pinecone.io`` will be used.
        :type host: str, optional
        :param proxy_url: The URL of the proxy to use for the connection.
        :type proxy_url: str, optional
        :param proxy_headers: Additional headers to pass to the proxy. Use this if your proxy setup requires authentication.
        :type proxy_headers: dict[str, str], optional
        :param ssl_ca_certs: The path to the SSL CA certificate bundle to use for the connection. This path should point to a file in PEM format. When not passed, the SDK will use the certificate bundle returned from ``certifi.where()``.
        :type ssl_ca_certs: str, optional
        :param ssl_verify: SSL verification is performed by default, but can be disabled using the boolean flag when testing with Pinecone Local or troubleshooting a proxy setup. You should never run with SSL verification disabled in production.
        :type ssl_verify: bool, optional
        :param additional_headers: Additional headers to pass to the API. This is mainly to support internal testing at Pinecone. End users should not need to use this unless following specific instructions to do so.
        :type additional_headers: dict[str, str], optional
        :param pool_threads: The number of threads to use for the ThreadPool when using methods that support the ``async_req`` keyword argument. The default number of threads is 5 * the number of CPUs in your execution environment.
        :type pool_threads: int, optional

        **Configuration with environment variables**

        If you instantiate the Pinecone client with no arguments, it will attempt to read the API key from the environment variable ``PINECONE_API_KEY``.

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

        **Configuration with keyword arguments**

        If you prefer being more explicit in your code, you can also pass the API key as a keyword argument. This is also where you will pass additional configuration options such as proxy settings if you wish to use those.

        .. code-block:: python

            import os
            from pinecone import Pinecone

            pc = Pinecone(
                api_key=os.environ.get("PINECONE_API_KEY"),
                host="https://api-staging.pinecone.io"
            )

        **Environment variables**

        The Pinecone client supports the following environment variables:

        * ``PINECONE_API_KEY``: The API key to use for authentication. If not passed via kwarg, the API key will be read from the environment variable ``PINECONE_API_KEY``.
        * ``PINECONE_DEBUG_CURL``: Enable some additional debug logging representing the HTTP requests as curl commands. The main use of is to run calls outside of the SDK to help evaluate whether a problem you are experiencing is due to the API's behavior or the behavior of the SDK itself.
        * ``PINECONE_ADDITIONAL_HEADERS``: A json string of a dictionary of header values to attach to all requests. This is primarily used for internal testing at Pinecone.

        .. warning::

            Be very careful with the ``PINECONE_DEBUG_CURL`` environment variable, as it will print out your API key which forms part of a required authentication header.

        **Proxy configuration**

        If your network setup requires you to interact with Pinecone via a proxy, you will need
        to pass additional configuration using optional keyword parameters. These optional parameters
        are forwarded to ``urllib3``, which is the underlying library currently used by the Pinecone client to
        make HTTP requests. You may find it helpful to refer to the
        `urllib3 documentation on working with proxies <https://urllib3.readthedocs.io/en/stable/advanced-usage.html#http-and-https-proxies>`_
        while troubleshooting these settings.

        Here is a basic example:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(
                api_key='YOUR_API_KEY',
                proxy_url='https://your-proxy.com'
            )

            pc.list_indexes()

        If your proxy requires authentication, you can pass those values in a header dictionary using the ``proxy_headers`` parameter.

        .. code-block:: python

            from pinecone import Pinecone
            from urllib3.util import make_headers

            pc = Pinecone(
                api_key='YOUR_API_KEY',
                proxy_url='https://your-proxy.com',
                proxy_headers=make_headers(proxy_basic_auth='username:password')
            )

            pc.list_indexes()

        **Using proxies with self-signed certificates**

        By default the Pinecone Python client will perform SSL certificate verification
        using the CA bundle maintained by Mozilla in the `certifi <https://pypi.org/project/certifi/>`_ package.
        If your proxy server is using a self-signed certificate, you will need to pass the path to the certificate
        in PEM format using the ``ssl_ca_certs`` parameter.

        .. code-block:: python

            from pinecone import Pinecone
            from urllib3.util import make_headers

            pc = Pinecone(
                api_key='YOUR_API_KEY',
                proxy_url='https://your-proxy.com',
                proxy_headers=make_headers(proxy_basic_auth='username:password'),
                ssl_ca_certs='path/to/cert-bundle.pem'
            )

            pc.list_indexes()

        **Disabling SSL verification**

        If you would like to disable SSL verification, you can pass the ``ssl_verify``
        parameter with a value of ``False``. We do not recommend going to production with SSL verification disabled.

        .. code-block:: python

            from pinecone import Pinecone
            from urllib3.util import make_headers

            pc = Pinecone(
                api_key='YOUR_API_KEY',
                proxy_url='https://your-proxy.com',
                proxy_headers=make_headers(proxy_basic_auth='username:password'),
                ssl_ca_certs='path/to/cert-bundle.pem',
                ssl_verify=False
            )

            pc.list_indexes()

        """
        for deprecated_kwarg in {"config", "openapi_config", "index_api"}:
            if deprecated_kwarg in kwargs:
                raise NotImplementedError(
                    f"Passing {deprecated_kwarg} is no longer supported. Please pass individual settings such as proxy_url, proxy_headers, ssl_ca_certs, and ssl_verify directly to the Pinecone constructor as keyword arguments. See the README at {docslinks['README']} for examples."
                )

        self._config = PineconeConfig.build(
            api_key=api_key,
            host=host,
            additional_headers=additional_headers,
            proxy_url=proxy_url,
            proxy_headers=proxy_headers,
            ssl_ca_certs=ssl_ca_certs,
            ssl_verify=ssl_verify,
            **kwargs,
        )
        """ :meta private: """

        self._openapi_config = ConfigBuilder.build_openapi_config(self._config, **kwargs)
        """ :meta private: """

        if pool_threads is None:
            self._pool_threads = 5 * cpu_count()
            """ :meta private: """
        else:
            self._pool_threads = pool_threads
            """ :meta private: """

        self._inference: "Inference" | None = None  # Lazy initialization
        """ :meta private: """

        self._db_control: "DBControl" | None = None  # Lazy initialization
        """ :meta private: """

        super().__init__()  # Initialize PluginAware

    @property
    def inference(self) -> "Inference":
        """
        Inference is a namespace where an instance of the `pinecone.inference.Inference` class is lazily created and cached.

        This property provides access to Pinecone's inference functionality, including embedding and reranking operations.

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")

            # Generate embeddings for text
            embeddings = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=["Disease prevention", "Immune system health"]
            )

            # Rerank documents based on query relevance
            reranked = pc.inference.rerank(
                model="bge-reranker-v2-m3",
                query="Disease prevention",
                documents=[
                    "Rich in vitamin C and other antioxidants, apples contribute to immune health and may reduce the risk of chronic diseases.",
                    "The high fiber content in apples can also help regulate blood sugar levels, making them beneficial for diabetes management.",
                    "Apples are a popular fruit known for their sweetness and crisp texture.",
                    "Regular exercise and a balanced diet are key components of maintaining good health and preventing illness.",
                ],
                top_n=2,
                rank_fields=["text"]
            )

        """
        if self._inference is None:
            from pinecone.inference import Inference

            self._inference = Inference(
                config=self._config,
                openapi_config=self._openapi_config,
                pool_threads=self._pool_threads,
            )
        return self._inference

    @property
    def db(self) -> "DBControl":
        """
        DBControl is a namespace where an instance of the `pinecone.db_control.DBControl` class is lazily created and cached.

        This property provides access to database control operations such as managing indexes, collections, and backups.

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")

            # Access database control operations
            indexes = pc.db.index.list()
            collections = pc.db.collection.list()
            backups = pc.db.backup.list()

        """
        if self._db_control is None:
            from pinecone.db_control import DBControl

            self._db_control = DBControl(
                config=self._config,
                openapi_config=self._openapi_config,
                pool_threads=self._pool_threads,
            )
        return self._db_control

    @property
    def index_host_store(self) -> "IndexHostStore":
        """:meta private:"""
        warnings.warn(
            "The `index_host_store` property is deprecated. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.db.index._index_host_store

    @property
    def config(self) -> "Config":
        """:meta private:"""
        # The config property is considered private, but the name cannot be changed to include underscore
        # without breaking compatibility with plugins in the wild.
        return self._config

    @property
    def openapi_config(self) -> "OpenApiConfiguration":
        """:meta private:"""
        warnings.warn(
            "The `openapi_config` property has been renamed to `_openapi_config`. It is considered private and should not be used directly. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._openapi_config

    @property
    def pool_threads(self) -> int:
        """:meta private:"""
        warnings.warn(
            "The `pool_threads` property has been renamed to `_pool_threads`. It is considered private and should not be used directly. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._pool_threads

    @property
    def index_api(self) -> "ManageIndexesApi":
        """:meta private:"""
        warnings.warn(
            "The `index_api` property is deprecated. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.db._index_api

    def create_index(
        self,
        name: str,
        spec: Dict | "ServerlessSpec" | "PodSpec" | "ByocSpec",
        dimension: int | None = None,
        metric: ("Metric" | str) | None = "cosine",
        timeout: int | None = None,
        deletion_protection: ("DeletionProtection" | str) | None = "disabled",
        vector_type: ("VectorType" | str) | None = "dense",
        tags: dict[str, str] | None = None,
    ) -> "IndexModel":
        """Creates a Pinecone index.

        :param name: The name of the index to create. Must be unique within your project and
            cannot be changed once created. Allowed characters are lowercase letters, numbers,
            and hyphens and the name may not begin or end with hyphens. Maximum length is 45 characters.
        :type name: str
        :param metric: Type of similarity metric used in the vector index when querying, one of ``{"cosine", "dotproduct", "euclidean"}``.
        :type metric: str, optional
        :param spec: A dictionary containing configurations describing how the index should be deployed. For serverless indexes,
            specify region and cloud. Optionally, you can specify ``read_capacity`` to configure dedicated read capacity mode
            (OnDemand or Dedicated) and ``schema`` to configure which metadata fields are filterable. For pod indexes, specify
            replicas, shards, pods, pod_type, metadata_config, and source_collection.
            Alternatively, use the ``ServerlessSpec``, ``PodSpec``, or ``ByocSpec`` objects to specify these configurations.
        :type spec: Dict
        :param dimension: If you are creating an index with ``vector_type="dense"`` (which is the default), you need to specify ``dimension`` to indicate the size of your vectors.
            This should match the dimension of the embeddings you will be inserting. For example, if you are using
            OpenAI's CLIP model, you should use ``dimension=1536``. Dimension is a required field when
            creating an index with ``vector_type="dense"`` and should not be passed when ``vector_type="sparse"``.
        :type dimension: int
        :type timeout: int, optional
        :param timeout: Specify the number of seconds to wait until index gets ready. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait.
        :param deletion_protection: If enabled, the index cannot be deleted. If disabled, the index can be deleted.
        :type deletion_protection: Optional[Literal["enabled", "disabled"]]
        :param vector_type: The type of vectors to be stored in the index. One of ``{"dense", "sparse"}``.
        :type vector_type: str, optional
        :param tags: Tags are key-value pairs you can attach to indexes to better understand, organize, and identify your resources. Some example use cases include tagging indexes with the name of the model that generated the embeddings, the date the index was created, or the purpose of the index.
        :type tags: Optional[dict[str, str]]
        :return: A ``IndexModel`` instance containing a description of the index that was created.

        Examples:

        .. code-block:: python
            :caption: Creating a serverless index

            import os
            from pinecone import (
                Pinecone,
                ServerlessSpec,
                CloudProvider,
                AwsRegion,
                Metric,
                DeletionProtection,
                VectorType
            )

            pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

            pc.create_index(
                name="my_index",
                dimension=512,
                metric=Metric.COSINE,
                spec=ServerlessSpec(
                    cloud=CloudProvider.AWS,
                    region=AwsRegion.US_WEST_2,
                    read_capacity={
                        "mode": "Dedicated",
                        "dedicated": {
                            "node_type": "t1",
                            "scaling": "Manual",
                            "manual": {"shards": 2, "replicas": 2},
                        },
                    },
                    schema={
                        "genre": {"filterable": True},
                        "year": {"filterable": True},
                        "rating": {"filterable": True},
                    },
                ),
                deletion_protection=DeletionProtection.DISABLED,
                vector_type=VectorType.DENSE,
                tags={
                    "app": "movie-recommendations",
                    "env": "production"
                }
            )

        .. code-block:: python
            :caption: Creating a pod index

            import os
            from pinecone import (
                Pinecone,
                PodSpec,
                PodIndexEnvironment,
                PodType,
                Metric,
                DeletionProtection,
                VectorType
            )

            pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

            pc.create_index(
                name="my_index",
                dimension=1536,
                metric=Metric.COSINE,
                spec=PodSpec(
                    environment=PodIndexEnvironment.US_EAST4_GCP,
                    pod_type=PodType.P1_X1
                ),
                deletion_protection=DeletionProtection.DISABLED,
                tags={
                    "model": "clip",
                    "app": "image-search",
                    "env": "testing"
                }
            )

        """
        return self.db.index.create(
            name=name,
            spec=spec,
            dimension=dimension,
            metric=metric,
            timeout=timeout,
            deletion_protection=deletion_protection,
            vector_type=vector_type,
            tags=tags,
        )

    def create_index_for_model(
        self,
        name: str,
        cloud: "CloudProvider" | str,
        region: "AwsRegion" | "GcpRegion" | "AzureRegion" | str,
        embed: "IndexEmbed" | "CreateIndexForModelEmbedTypedDict",
        tags: dict[str, str] | None = None,
        deletion_protection: ("DeletionProtection" | str) | None = "disabled",
        read_capacity: (
            "ReadCapacityDict"
            | "ReadCapacity"
            | "ReadCapacityOnDemandSpec"
            | "ReadCapacityDedicatedSpec"
        )
        | None = None,
        schema: (
            dict[
                str, "MetadataSchemaFieldConfig"
            ]  # Direct field mapping: {field_name: {filterable: bool}}
            | dict[
                str, dict[str, Any]
            ]  # Dict with "fields" wrapper: {"fields": {field_name: {...}}, ...}
            | "BackupModelSchema"  # OpenAPI model instance
        )
        | None = None,
        timeout: int | None = None,
    ) -> "IndexModel":
        """Create a Serverless index configured for use with Pinecone's integrated inference models.

        :param name: The name of the index to create. Must be unique within your project and
            cannot be changed once created. Allowed characters are lowercase letters, numbers,
            and hyphens and the name may not begin or end with hyphens. Maximum length is 45 characters.
        :type name: str
        :param cloud: The cloud provider to use for the index. One of ``{"aws", "gcp", "azure"}``.
        :type cloud: str
        :param region: The region to use for the index. Enum objects ``AwsRegion``, ``GcpRegion``, and ``AzureRegion`` are also available to help you quickly set these parameters, but may not be up to date as new regions become available.
        :type region: str
        :param embed: The embedding configuration for the index. This param accepts a dictionary or an instance of the ``IndexEmbed`` object.
        :type embed: Union[Dict, IndexEmbed]
        :param tags: Tags are key-value pairs you can attach to indexes to better understand, organize, and identify your resources. Some example use cases include tagging indexes with the name of the model that generated the embeddings, the date the index was created, or the purpose of the index.
        :type tags: Optional[dict[str, str]]
        :param deletion_protection: If enabled, the index cannot be deleted. If disabled, the index can be deleted. This setting can be changed with ``configure_index``.
        :type deletion_protection: Optional[Literal["enabled", "disabled"]]
        :param read_capacity: Optional read capacity configuration. You can specify ``read_capacity`` to configure dedicated read capacity mode
            (OnDemand or Dedicated). See ``ServerlessSpec`` documentation for details on read capacity configuration.
        :type read_capacity: Optional[Union[ReadCapacityDict, ReadCapacity, ReadCapacityOnDemandSpec, ReadCapacityDedicatedSpec]]
        :param schema: Optional metadata schema configuration. You can specify ``schema`` to configure which metadata fields are filterable.
            The schema can be provided as a dictionary mapping field names to their configurations (e.g., ``{"genre": {"filterable": True}}``)
            or as a dictionary with a ``fields`` key (e.g., ``{"fields": {"genre": {"filterable": True}}}``).
        :type schema: Optional[Union[dict[str, MetadataSchemaFieldConfig], dict[str, dict[str, Any]], BackupModelSchema]]
        :type timeout: Optional[int]
        :param timeout: Specify the number of seconds to wait until index is ready to receive data. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait.
        :return: A description of the index that was created.
        :rtype: IndexModel

        The resulting index can be described, listed, configured, and deleted like any other Pinecone index with the ``describe_index``, ``list_indexes``, ``configure_index``, and ``delete_index`` methods.

        After the model is created, you can upsert records into the index with the ``upsert_records`` method, and search your records with the ``search`` method.

        .. code-block:: python

            from pinecone import (
                Pinecone,
                IndexEmbed,
                CloudProvider,
                AwsRegion,
                EmbedModel,
                Metric,
            )

            pc = Pinecone()

            if not pc.has_index("book-search"):
                desc = pc.create_index_for_model(
                    name="book-search",
                    cloud=CloudProvider.AWS,
                    region=AwsRegion.US_EAST_1,
                    embed=IndexEmbed(
                        model=EmbedModel.Multilingual_E5_Large,
                        metric=Metric.COSINE,
                        field_map={
                            "text": "description",
                        },
                    )
                )

        .. code-block:: python
            :caption: Creating an index for model with schema and dedicated read capacity

            from pinecone import (
                Pinecone,
                IndexEmbed,
                CloudProvider,
                AwsRegion,
                EmbedModel,
                Metric,
            )

            pc = Pinecone()

            if not pc.has_index("book-search"):
                desc = pc.create_index_for_model(
                    name="book-search",
                    cloud=CloudProvider.AWS,
                    region=AwsRegion.US_EAST_1,
                    embed=IndexEmbed(
                        model=EmbedModel.Multilingual_E5_Large,
                        metric=Metric.COSINE,
                        field_map={
                            "text": "description",
                        },
                    ),
                    read_capacity={
                        "mode": "Dedicated",
                        "dedicated": {
                            "node_type": "t1",
                            "scaling": "Manual",
                            "manual": {"shards": 2, "replicas": 2},
                        },
                    },
                    schema={
                        "genre": {"filterable": True},
                        "year": {"filterable": True},
                        "rating": {"filterable": True},
                    },
                )

        .. seealso::

            Official docs on `available cloud regions <https://docs.pinecone.io/troubleshooting/available-cloud-regions>`_

            `Model Gallery <https://docs.pinecone.io/models/overview>`_ to learn about available models

        """
        return self.db.index.create_for_model(
            name=name,
            cloud=cloud,
            region=region,
            embed=embed,
            tags=tags,
            deletion_protection=deletion_protection,
            read_capacity=read_capacity,
            schema=schema,
            timeout=timeout,
        )

    @require_kwargs
    def create_index_from_backup(
        self,
        *,
        name: str,
        backup_id: str,
        deletion_protection: ("DeletionProtection" | str) | None = "disabled",
        tags: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> "IndexModel":
        """Create an index from a backup.

        Call ``list_backups`` to get a list of backups for your project.

        :param name: The name of the index to create.
        :type name: str
        :param backup_id: The ID of the backup to restore.
        :type backup_id: str
        :param deletion_protection: If enabled, the index cannot be deleted. If disabled, the index can be deleted. This setting can be changed with ``configure_index``.
        :type deletion_protection: Optional[Literal["enabled", "disabled"]]
        :param tags: Tags are key-value pairs you can attach to indexes to better understand, organize, and identify your resources. Some example use cases include tagging indexes with the name of the model that generated the embeddings, the date the index was created, or the purpose of the index.
        :type tags: Optional[dict[str, str]]
        :param timeout: Specify the number of seconds to wait until index is ready to receive data. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait.
        :return: A description of the index that was created.
        :rtype: IndexModel

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            # List available backups
            backups = pc.list_backups()
            if backups:
                backup_id = backups[0].id

                # Create index from backup
                index = pc.create_index_from_backup(
                    name="restored-index",
                    backup_id=backup_id,
                    deletion_protection="disabled"
                )

        """
        return self.db.index.create_from_backup(
            name=name,
            backup_id=backup_id,
            deletion_protection=deletion_protection,
            tags=tags,
            timeout=timeout,
        )

    def delete_index(self, name: str, timeout: int | None = None) -> None:
        """Deletes a Pinecone index.

        :param name: the name of the index.
        :type name: str
        :param timeout: Number of seconds to poll status checking whether the index has been deleted. If None,
            wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait.
        :type timeout: int, optional

        Deleting an index is an irreversible operation. All data in the index will be lost.
        When you use this command, a request is sent to the Pinecone control plane to delete
        the index, but the termination is not synchronous because resources take a few moments to
        be released.

        By default the ``delete_index`` method will block until polling of the ``describe_index`` method
        shows that the delete operation has completed. If you prefer to return immediately and not
        wait for the index to be deleted, you can pass ``timeout=-1`` to the method.

        After the delete request is submitted, polling ``describe_index`` will show that the index
        transitions into a ``Terminating`` state before eventually resulting in a 404 after it has been removed.

        This operation can fail if the index is configured with ``deletion_protection="enabled"``.
        In this case, you will need to call ``configure_index`` to disable deletion protection before
        you can delete the index.

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            index_name = "my_index"
            desc = pc.describe_index(name=index_name)

            if desc.deletion_protection == "enabled":
                # If for some reason deletion protection is enabled, you will need to disable it first
                # before you can delete the index. But use caution as this operation is not reversible
                # and if somebody enabled deletion protection, they probably had a good reason.
                pc.configure_index(name=index_name, deletion_protection="disabled")

            pc.delete_index(name=index_name)

        """
        return self.db.index.delete(name=name, timeout=timeout)

    def list_indexes(self) -> "IndexList":
        """Lists all indexes in your project.

        :return: Returns an ``IndexList`` object, which is iterable and contains a
            list of ``IndexModel`` objects. The ``IndexList`` also has a convenience method ``names()``
            which returns a list of index names for situations where you just want to iterate over
            all index names.

        The results include a description of all indexes in your project, including the
        index name, dimension, metric, status, and spec.

        If you simply want to check whether an index exists, see the ``has_index()`` convenience method.

        You can use the ``list_indexes()`` method to iterate over descriptions of every index in your project.

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            for index in pc.list_indexes():
                print(index.name)
                print(index.dimension)
                print(index.metric)
                print(index.status)
                print(index.host)
                print(index.spec)

        """
        return self.db.index.list()

    def describe_index(self, name: str) -> "IndexModel":
        """Describes a Pinecone index.

        :param name: the name of the index to describe.
        :return: Returns an ``IndexModel`` object
          which gives access to properties such as the
          index name, dimension, metric, host url, status,
          and spec.

        **Getting your index host url**

        In a real production situation, you probably want to
        store the host url in an environment variable so you
        don't have to call describe_index and re-fetch it
        every time you want to use the index. But this example
        shows how to get the value from the API using describe_index.

        .. code-block:: python

            from pinecone import Pinecone, Index

            pc = Pinecone()

            index_name="my_index"
            description = pc.describe_index(name=index_name)
            print(description)
            # {
            #     "name": "my_index",
            #     "metric": "cosine",
            #     "host": "my_index-dojoi3u.svc.aped-4627-b74a.pinecone.io",
            #     "spec": {
            #         "serverless": {
            #             "cloud": "aws",
            #             "region": "us-east-1"
            #         }
            #     },
            #     "status": {
            #         "ready": true,
            #         "state": "Ready"
            #     },
            #     "vector_type": "dense",
            #     "dimension": 1024,
            #     "deletion_protection": "enabled",
            #     "tags": {
            #         "environment": "production"
            #     }
            # }

            print(f"Your index is hosted at {description.host}")

            index = pc.Index(host=description.host)
            index.upsert(vectors=[...])

        """
        return self.db.index.describe(name=name)

    def has_index(self, name: str) -> bool:
        """Checks if a Pinecone index exists.

        :param name: The name of the index to check for existence.
        :return: Returns ``True`` if the index exists, ``False`` otherwise.

        .. code-block:: python

            from pinecone import Pinecone, ServerlessSpec

            pc = Pinecone()

            index_name = "my_index"
            if not pc.has_index(index_name):
                print("Index does not exist, creating...")
                pc.create_index(
                    name=index_name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-west-2")
                )

        """
        return self.db.index.has(name=name)

    def configure_index(
        self,
        name: str,
        replicas: int | None = None,
        pod_type: ("PodType" | str) | None = None,
        deletion_protection: ("DeletionProtection" | str) | None = None,
        tags: dict[str, str] | None = None,
        embed: ("ConfigureIndexEmbed" | Dict) | None = None,
        read_capacity: (
            "ReadCapacityDict"
            | "ReadCapacity"
            | "ReadCapacityOnDemandSpec"
            | "ReadCapacityDedicatedSpec"
        )
        | None = None,
    ) -> None:
        """Modify an index's configuration.

        :param name: the name of the Index
        :type name: str, required
        :param replicas: the desired number of replicas, lowest value is 0.
        :type replicas: int, optional
        :param pod_type: the new ``pod_type`` for the index. To learn more about the
            available pod types, please see `Understanding Indexes <https://docs.pinecone.io/docs/indexes>`_.
            Note that pod type is only available for pod-based indexes.
        :type pod_type: str or PodType, optional
        :param deletion_protection: If set to ``'enabled'``, the index cannot be deleted. If ``'disabled'``, the index can be deleted.
        :type deletion_protection: str or DeletionProtection, optional
        :param tags: A dictionary of tags to apply to the index. Tags are key-value pairs that can be used to organize and manage indexes. To remove a tag, set the value to "". Tags passed to configure_index will be merged with existing tags and any with the value empty string will be removed.
        :type tags: dict[str, str], optional
        :param embed: configures the integrated inference embedding settings for the index. You can convert an existing index to an integrated index by specifying the embedding model and field_map.
            The index vector type and dimension must match the model vector type and dimension, and the index similarity metric must be supported by the model.
            You can later change the embedding configuration to update the field_map, read_parameters, or write_parameters. Once set, the model cannot be changed.
        :type embed: Optional[Union[ConfigureIndexEmbed, Dict]], optional
        :param read_capacity: Optional read capacity configuration for serverless indexes. You can specify ``read_capacity`` to configure dedicated read capacity mode
            (OnDemand or Dedicated). See ``ServerlessSpec`` documentation for details on read capacity configuration.
            Note that read capacity configuration is only available for serverless indexes.
        :type read_capacity: Optional[Union[ReadCapacityDict, ReadCapacity, ReadCapacityOnDemandSpec, ReadCapacityDedicatedSpec]]

        This method is used to modify an index's configuration. It can be used to:

        * Configure read capacity for serverless indexes using ``read_capacity``
        * Scale a pod-based index horizontally using ``replicas``
        * Scale a pod-based index vertically using ``pod_type``
        * Enable or disable deletion protection using ``deletion_protection``
        * Add, change, or remove tags using ``tags``

        **Configuring read capacity for serverless indexes**

        To configure read capacity for serverless indexes, pass the ``read_capacity`` parameter to the ``configure_index`` method.
        You can configure either OnDemand or Dedicated read capacity mode.

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            # Configure to OnDemand read capacity (default)
            pc.configure_index(
                name="my_index",
                read_capacity={"mode": "OnDemand"}
            )

            # Configure to Dedicated read capacity with manual scaling
            pc.configure_index(
                name="my_index",
                read_capacity={
                    "mode": "Dedicated",
                    "dedicated": {
                        "node_type": "t1",
                        "scaling": "Manual",
                        "manual": {"shards": 1, "replicas": 1}
                    }
                }
            )

            # Verify the configuration was applied
            desc = pc.describe_index("my_index")
            assert desc.spec.serverless.read_capacity.mode == "Dedicated"

        **Scaling pod-based indexes**

        To scale your pod-based index, you pass a ``replicas`` and/or ``pod_type`` param to the ``configure_index`` method. ``pod_type`` may be a string or a value from the ``PodType`` enum.

        .. code-block:: python

            from pinecone import Pinecone, PodType

            pc = Pinecone()
            pc.configure_index(
                name="my_index",
                replicas=2,
                pod_type=PodType.P1_X2
            )

        After providing these new configurations, you must call ``describe_index`` to see the status of the index as the changes are applied.

        **Enabling or disabling deletion protection**

        To enable or disable deletion protection, pass the ``deletion_protection`` parameter to the ``configure_index`` method. When deletion protection
        is enabled, the index cannot be deleted with the ``delete_index`` method.

        .. code-block:: python

            from pinecone import Pinecone, DeletionProtection

            pc = Pinecone()

            # Enable deletion protection
            pc.configure_index(
                name="my_index",
                deletion_protection=DeletionProtection.ENABLED
            )

            # Call describe_index to see the change was applied.
            assert pc.describe_index("my_index").deletion_protection == "enabled"

            # Disable deletion protection
            pc.configure_index(
                name="my_index",
                deletion_protection=DeletionProtection.DISABLED
            )

        **Adding, changing, or removing tags**

        To add, change, or remove tags, pass the ``tags`` parameter to the ``configure_index`` method. When tags are passed using ``configure_index``,
        they are merged with any existing tags already on the index. To remove a tag, set the value of the key to an empty string.

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            # Add a tag
            pc.configure_index(name="my_index", tags={"environment": "staging"})

            # Change a tag
            pc.configure_index(name="my_index", tags={"environment": "production"})

            # Remove a tag
            pc.configure_index(name="my_index", tags={"environment": ""})

            # Call describe_index to view the tags are changed
            print(pc.describe_index("my_index").tags)

        """
        return self.db.index.configure(
            name=name,
            replicas=replicas,
            pod_type=pod_type,
            deletion_protection=deletion_protection,
            tags=tags,
            embed=embed,
            read_capacity=read_capacity,
        )

    def create_collection(self, name: str, source: str) -> None:
        """Create a collection from a pod-based index.

        :param name: Name of the collection
        :type name: str, required
        :param source: Name of the source index
        :type source: str, required

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            # Create a collection from an existing pod-based index
            pc.create_collection(name="my_collection", source="my_index")

        """
        return self.db.collection.create(name=name, source=source)

    def list_collections(self) -> "CollectionList":
        """List all collections.

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            for collection in pc.list_collections():
                print(collection.name)
                print(collection.source)

            # You can also iterate specifically over the collection
            # names with the .names() helper.
            collection_name="my_collection"
            for collection_name in pc.list_collections().names():
                print(collection_name)

        """
        return self.db.collection.list()

    def delete_collection(self, name: str) -> None:
        """Deletes a collection.

        :param str name: The name of the collection to delete.

        Deleting a collection is an irreversible operation. All data
        in the collection will be lost.

        This method tells Pinecone you would like to delete a collection,
        but it takes a few moments to complete the operation. Use the
        ``describe_collection()`` method to confirm that the collection
        has been deleted.

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            pc.delete_collection(name="my_collection")

        """
        return self.db.collection.delete(name=name)

    def describe_collection(self, name: str) -> dict[str, Any]:
        """Describes a collection.

        :param str name: The name of the collection

        :return: Description of the collection

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            description = pc.describe_collection("my_collection")
            print(description.name)
            print(description.source)
            print(description.status)
            print(description.size)

        """
        from typing import cast

        result = self.db.collection.describe(name=name)
        return cast(dict[str, Any], result)

    @require_kwargs
    def create_backup(
        self, *, index_name: str, backup_name: str, description: str = ""
    ) -> "BackupModel":
        """Create a backup of an index.

        :param index_name: The name of the index to backup.
        :type index_name: str
        :param backup_name: The name to give the backup.
        :type backup_name: str
        :param description: Optional description of the backup.
        :type description: str, optional

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            # Create a backup of an index
            backup = pc.create_backup(
                index_name="my_index",
                backup_name="my_backup",
                description="Daily backup"
            )

            print(f"Backup created with ID: {backup.id}")

        """
        return self.db.backup.create(
            index_name=index_name, backup_name=backup_name, description=description
        )

    @require_kwargs
    def list_backups(
        self,
        *,
        index_name: str | None = None,
        limit: int | None = 10,
        pagination_token: str | None = None,
    ) -> "BackupList":
        """List backups.

        If ``index_name`` is provided, the backups will be filtered by index. If no ``index_name`` is provided, all backups in the project will be returned.

        :param index_name: The name of the index to list backups for.
        :type index_name: str, optional
        :param limit: The maximum number of backups to return.
        :type limit: int, optional
        :param pagination_token: The pagination token to use for pagination.
        :type pagination_token: str, optional

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            # List all backups
            all_backups = pc.list_backups(limit=20)

            # List backups for a specific index
            index_backups = pc.list_backups(index_name="my_index", limit=10)

            for backup in index_backups:
                print(f"Backup: {backup.name}, Status: {backup.status}")

        """
        return self.db.backup.list(
            index_name=index_name, limit=limit, pagination_token=pagination_token
        )

    @require_kwargs
    def describe_backup(self, *, backup_id: str) -> "BackupModel":
        """Describe a backup.

        :param backup_id: The ID of the backup to describe.
        :type backup_id: str

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            backup = pc.describe_backup(backup_id="backup-123")
            print(f"Backup: {backup.name}")
            print(f"Status: {backup.status}")
            print(f"Index: {backup.index_name}")

        """
        return self.db.backup.describe(backup_id=backup_id)

    @require_kwargs
    def delete_backup(self, *, backup_id: str) -> None:
        """Delete a backup.

        :param backup_id: The ID of the backup to delete.
        :type backup_id: str

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            pc.delete_backup(backup_id="backup-123")

        """
        return self.db.backup.delete(backup_id=backup_id)

    @require_kwargs
    def list_restore_jobs(
        self, *, limit: int | None = 10, pagination_token: str | None = None
    ) -> "RestoreJobList":
        """List restore jobs.

        :param limit: The maximum number of restore jobs to return.
        :type limit: int
        :param pagination_token: The pagination token to use for pagination.
        :type pagination_token: str

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            restore_jobs = pc.list_restore_jobs(limit=20)

            for job in restore_jobs:
                print(f"Job ID: {job.id}, Status: {job.status}")

        """
        return self.db.restore_job.list(limit=limit, pagination_token=pagination_token)

    @require_kwargs
    def describe_restore_job(self, *, job_id: str) -> "RestoreJobModel":
        """Describe a restore job.

        :param job_id: The ID of the restore job to describe.
        :type job_id: str

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            job = pc.describe_restore_job(job_id="job-123")
            print(f"Job ID: {job.id}")
            print(f"Status: {job.status}")
            print(f"Source backup: {job.backup_id}")

        """
        return self.db.restore_job.describe(job_id=job_id)

    @staticmethod
    def from_texts(*args: Any, **kwargs: Any) -> NoReturn:
        """:meta private:"""
        raise AttributeError(_build_langchain_attribute_error_message("from_texts"))

    @staticmethod
    def from_documents(*args: Any, **kwargs: Any) -> NoReturn:
        """:meta private:"""
        raise AttributeError(_build_langchain_attribute_error_message("from_documents"))

    def Index(self, name: str = "", host: str = "", **kwargs) -> "Index":
        """Target an index for data operations.

        :param name: The name of the index to target. If you specify the name of the index, the client will
            fetch the host url from the Pinecone control plane.
        :type name: str, optional
        :param host: The host url of the index to target. If you specify the host url, the client will use
            the host url directly without making any additional calls to the control plane.
        :type host: str, optional
        :param pool_threads: The number of threads to use when making parallel requests by calling index methods with optional kwarg async_req=True, or using methods that make use of thread-based parallelism automatically such as query_namespaces().
        :type pool_threads: int, optional
        :param connection_pool_maxsize: The maximum number of connections to keep in the connection pool.
        :type connection_pool_maxsize: int, optional
        :return: An instance of the ``Index`` class.

        **Target an index by host url**

        In production situations, you want to upsert or query your data as quickly
        as possible. If you know in advance the host url of your index, you can
        eliminate a round trip to the Pinecone control plane by specifying the
        host of the index. If instead you pass the name of the index, the client
        will need to make an additional call to api.pinecone.io to get the host url
        before any data operations can take place.

        .. code-block:: python

            import os
            from pinecone import Pinecone

            api_key = os.environ.get("PINECONE_API_KEY")
            index_host = os.environ.get("PINECONE_INDEX_HOST")

            pc = Pinecone(api_key=api_key)
            index = pc.Index(host=index_host)

            # Now you're ready to perform data operations
            index.query(vector=[...], top_k=10)

        To find your host url, you can use the describe_index method to call api.pinecone.io.
        The host url is returned in the response. Or, alternatively, the
        host is displayed in the Pinecone web console.

        .. code-block:: python

            import os
            from pinecone import Pinecone

            pc = Pinecone(
                api_key=os.environ.get("PINECONE_API_KEY")
            )

            host = pc.describe_index('index-name').host

        **Target an index by name (not recommended for production)**

        For more casual usage, such as when you are playing and exploring with Pinecone
        in a notebook setting, you can also target an index by name. If you use this
        approach, the client may need to perform an extra call to the Pinecone control
        plane to get the host url on your behalf to get the index host.

        The client will cache the index host for future use whenever it is seen, so you
        will only incur the overhead of only one call. But this approach is not
        recommended for production usage because it introduces an unnecessary runtime
        dependency on api.pinecone.io.

        .. code-block:: python

            import os
            from pinecone import Pinecone, ServerlessSpec

            api_key = os.environ.get("PINECONE_API_KEY")

            pc = Pinecone(api_key=api_key)
            pc.create_index(
                name='my_index',
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-west-2')
            )
            index = pc.Index('my_index')

            # Now you're ready to perform data operations
            index.query(vector=[...], top_k=10)

        """
        from pinecone.db_data import _Index

        if name == "" and host == "":
            raise ValueError("Either name or host must be specified")

        pt = kwargs.pop("pool_threads", None) or self._pool_threads
        api_key = self._config.api_key
        openapi_config = self._openapi_config

        if host != "":
            check_realistic_host(host)

            # Use host url if it is provided
            index_host = normalize_host(host)
        else:
            # Otherwise, get host url from describe_index using the index name
            index_host = self.db.index._get_host(name)

        return _Index(
            host=index_host,
            api_key=api_key,
            pool_threads=pt,
            openapi_config=openapi_config,
            source_tag=self.config.source_tag,
            **kwargs,
        )

    def IndexAsyncio(self, host: str, **kwargs) -> "IndexAsyncio":
        """Build an asyncio-compatible Index object.

        :param host: The host url of the index to target. You can find this url in the Pinecone
            web console or by calling describe_index method of ``Pinecone`` or ``PineconeAsyncio``.
        :type host: str, required

        :return: An instance of the ``IndexAsyncio`` class.

        .. code-block:: python

            import asyncio
            import os
            from pinecone import Pinecone

            async def main():
                pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
                async with pc.IndexAsyncio(host=os.environ.get("PINECONE_INDEX_HOST")) as index:
                    await index.query(vector=[...], top_k=10)

            asyncio.run(main())

        See more docs for ``PineconeAsyncio`` `here <./asyncio.html#db-data-plane>`_.

        """
        from pinecone.db_data import _IndexAsyncio

        api_key = self._config.api_key
        openapi_config = self._openapi_config

        if host is None or host == "":
            raise ValueError("A host must be specified")

        check_realistic_host(host)
        index_host = normalize_host(host)

        return _IndexAsyncio(
            host=index_host,
            api_key=api_key,
            openapi_config=openapi_config,
            source_tag=self.config.source_tag,
            **kwargs,
        )


def check_realistic_host(host: str) -> None:
    """:meta private:

    Checks whether a user-provided host string seems plausible.
    Someone could erroneously pass an index name as the host by
    mistake, and if they have done that we'd like to give them a
    simple error message as feedback rather than attempting to
    call the url and getting a more cryptic DNS resolution error.
    """

    if "." not in host and "localhost" not in host:
        raise ValueError(
            f"You passed '{host}' as the host but this does not appear to be valid. Call describe_index() to confirm the host of the index."
        )
