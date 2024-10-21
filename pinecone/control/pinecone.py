import time
import logging
from typing import Optional, Dict, Any, Union, Literal

from .index_host_store import IndexHostStore

from pinecone.config import PineconeConfig, Config, ConfigBuilder

from pinecone.core.openapi.control.api.manage_indexes_api import ManageIndexesApi
from pinecone.core.openapi.shared.api_client import ApiClient


from pinecone.utils import (
    normalize_host,
    setup_openapi_client,
    build_plugin_setup_client,
    parse_non_empty_args,
)
from pinecone.core.openapi.control.models import (
    CreateCollectionRequest,
    CreateIndexRequest,
    ConfigureIndexRequest,
    ConfigureIndexRequestSpec,
    ConfigureIndexRequestSpecPod,
    DeletionProtection,
    IndexSpec,
    ServerlessSpec as ServerlessSpecModel,
    PodSpec as PodSpecModel,
    PodSpecMetadataConfig,
)
from pinecone.core.openapi.shared import API_VERSION
from pinecone.models import ServerlessSpec, PodSpec, IndexModel, IndexList, CollectionList
from .langchain_import_warnings import _build_langchain_attribute_error_message

from pinecone.data import Index

from pinecone_plugin_interface import load_and_install as install_plugins

logger = logging.getLogger(__name__)


class Pinecone:
    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        proxy_url: Optional[str] = None,
        proxy_headers: Optional[Dict[str, str]] = None,
        ssl_ca_certs: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        config: Optional[Config] = None,
        additional_headers: Optional[Dict[str, str]] = {},
        pool_threads: Optional[int] = 1,
        index_api: Optional[ManageIndexesApi] = None,
        **kwargs,
    ):
        """
        The `Pinecone` class is the main entry point for interacting with Pinecone via this Python SDK.
        It is used to create, delete, and manage your indexes and collections.

        :param api_key: The API key to use for authentication. If not passed via kwarg, the API key will be read from the environment variable `PINECONE_API_KEY`.
        :type api_key: str, optional
        :param host: The control plane host to connect to.
        :type host: str, optional
        :param proxy_url: The URL of the proxy to use for the connection. Default: `None`
        :type proxy_url: str, optional
        :param proxy_headers: Additional headers to pass to the proxy. Use this if your proxy setup requires authentication. Default: `{}`
        :type proxy_headers: Dict[str, str], optional
        :param ssl_ca_certs: The path to the SSL CA certificate bundle to use for the connection. This path should point to a file in PEM format. Default: `None`
        :type ssl_ca_certs: str, optional
        :param ssl_verify: SSL verification is performed by default, but can be disabled using the boolean flag. Default: `True`
        :type ssl_verify: bool, optional
        :param config: A `pinecone.config.Config` object. If passed, the `api_key` and `host` parameters will be ignored.
        :type config: pinecone.config.Config, optional
        :param additional_headers: Additional headers to pass to the API. Default: `{}`
        :type additional_headers: Dict[str, str], optional
        :param pool_threads: The number of threads to use for the connection pool. Default: `1`
        :type pool_threads: int, optional
        :param index_api: An instance of `pinecone.core.client.api.manage_indexes_api.ManageIndexesApi`. If passed, the `host` parameter will be ignored.
        :type index_api: pinecone.core.client.api.manage_indexes_api.ManageIndexesApi, optional


        ### Configuration with environment variables

        If you instantiate the Pinecone client with no arguments, it will attempt to read the API key from the environment variable `PINECONE_API_KEY`.

        ```python
        from pinecone import Pinecone

        pc = Pinecone()
        ```

        ### Configuration with keyword arguments

        If you prefer being more explicit in your code, you can also pass the API  as a keyword argument.

        ```python
        import os
        from pinecone import Pinecone

        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        ```

        ### Environment variables

        The Pinecone client supports the following environment variables:

        - `PINECONE_API_KEY`: The API key to use for authentication. If not passed via
        kwarg, the API key will be read from the environment variable `PINECONE_API_KEY`.

        - `PINECONE_DEBUG_CURL`: When troubleshooting it can be very useful to run curl
        commands against the control plane API to see exactly what data is being sent
        and received without all the abstractions and transformations applied by the Python
        SDK. If you set this environment variable to `true`, the Pinecone client will use
        request parameters to print out an equivalent curl command that you can run yourself
        or share with Pinecone support. **Be very careful with this option, as it will print out
        your API key** which forms part of a required authentication header. Default: `false`

        ### Proxy configuration

        If your network setup requires you to interact with Pinecone via a proxy, you will need
        to pass additional configuration using optional keyword parameters. These optional parameters
        are forwarded to `urllib3`, which is the underlying library currently used by the Pinecone client to
        make HTTP requests. You may find it helpful to refer to the
        [urllib3 documentation on working with proxies](https://urllib3.readthedocs.io/en/stable/advanced-usage.html#http-and-https-proxies)
        while troubleshooting these settings.

        Here is a basic example:

        ```python
        from pinecone import Pinecone

        pc = Pinecone(
            api_key='YOUR_API_KEY',
            proxy_url='https://your-proxy.com'
        )

        pc.list_indexes()
        ```

        If your proxy requires authentication, you can pass those values in a header dictionary using the `proxy_headers` parameter.

        ```python
        from pinecone import Pinecone
        import urllib3 import make_headers

        pc = Pinecone(
            api_key='YOUR_API_KEY',
            proxy_url='https://your-proxy.com',
            proxy_headers=make_headers(proxy_basic_auth='username:password')
        )

        pc.list_indexes()
        ```

        ### Using proxies with self-signed certificates

        By default the Pinecone Python client will perform SSL certificate verification
        using the CA bundle maintained by Mozilla in the [certifi](https://pypi.org/project/certifi/) package.
        If your proxy server is using a self-signed certificate, you will need to pass the path to the certificate
        in PEM format using the `ssl_ca_certs` parameter.

        ```python
        from pinecone import Pinecone
        import urllib3 import make_headers

        pc = Pinecone(
            api_key='YOUR_API_KEY',
            proxy_url='https://your-proxy.com',
            proxy_headers=make_headers(proxy_basic_auth='username:password'),
            ssl_ca_certs='path/to/cert-bundle.pem'
        )

        pc.list_indexes()
        ```

        ### Disabling SSL verification

        If you would like to disable SSL verification, you can pass the `ssl_verify`
        parameter with a value of `False`. We do not recommend going to production with SSL verification disabled.

        ```python
        from pinecone import Pinecone
        import urllib3 import make_headers

        pc = Pinecone(
            api_key='YOUR_API_KEY',
            proxy_url='https://your-proxy.com',
            proxy_headers=make_headers(proxy_basic_auth='username:password'),
            ssl_ca_certs='path/to/cert-bundle.pem',
            ssl_verify=False
        )

        pc.list_indexes()

        ```
        """
        if config:
            if not isinstance(config, Config):
                raise TypeError("config must be of type pinecone.config.Config")
            else:
                self.config = config
        else:
            self.config = PineconeConfig.build(
                api_key=api_key,
                host=host,
                additional_headers=additional_headers,
                proxy_url=proxy_url,
                proxy_headers=proxy_headers,
                ssl_ca_certs=ssl_ca_certs,
                ssl_verify=ssl_verify,
                **kwargs,
            )

        if kwargs.get("openapi_config", None):
            raise Exception(
                "Passing openapi_config is no longer supported. Please pass settings such as proxy_url, proxy_headers, ssl_ca_certs, and ssl_verify directly to the Pinecone constructor as keyword arguments. See the README at https://github.com/pinecone-io/pinecone-python-client for examples."
            )

        self.openapi_config = ConfigBuilder.build_openapi_config(self.config, **kwargs)
        self.pool_threads = pool_threads

        if index_api:
            self.index_api = index_api
        else:
            self.index_api = setup_openapi_client(
                api_client_klass=ApiClient,
                api_klass=ManageIndexesApi,
                config=self.config,
                openapi_config=self.openapi_config,
                pool_threads=pool_threads,
                api_version=API_VERSION,
            )

        self.index_host_store = IndexHostStore()
        """ @private """

        self.load_plugins()

    def load_plugins(self):
        """@private"""
        try:
            # I don't expect this to ever throw, but wrapping this in a
            # try block just in case to make sure a bad plugin doesn't
            # halt client initialization.
            openapi_client_builder = build_plugin_setup_client(
                config=self.config,
                openapi_config=self.openapi_config,
                pool_threads=self.pool_threads,
            )
            install_plugins(self, openapi_client_builder)
        except Exception as e:
            logger.error(f"Error loading plugins: {e}")

    def create_index(
        self,
        name: str,
        dimension: int,
        spec: Union[Dict, ServerlessSpec, PodSpec],
        metric: Optional[str] = "cosine",
        timeout: Optional[int] = None,
        deletion_protection: Optional[Literal["enabled", "disabled"]] = "disabled",
    ):
        """Creates a Pinecone index.

        :param name: The name of the index to create. Must be unique within your project and
            cannot be changed once created. Allowed characters are lowercase letters, numbers,
            and hyphens and the name may not begin or end with hyphens. Maximum length is 45 characters.
        :type name: str
        :param dimension: The dimension of vectors that will be inserted in the index. This should
            match the dimension of the embeddings you will be inserting. For example, if you are using
            OpenAI's CLIP model, you should use `dimension=1536`.
        :type dimension: int
        :param metric: Type of metric used in the vector index when querying, one of `{"cosine", "dotproduct", "euclidean"}`. Defaults to `"cosine"`.
            Defaults to `"cosine"`.
        :type metric: str, optional
        :param spec: A dictionary containing configurations describing how the index should be deployed. For serverless indexes,
            specify region and cloud. For pod indexes, specify replicas, shards, pods, pod_type, metadata_config, and source_collection.
        :type spec: Dict
        :type timeout: int, optional
        :param timeout: Specify the number of seconds to wait until index gets ready. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait. Default: None
        :param deletion_protection: If enabled, the index cannot be deleted. If disabled, the index can be deleted. Default: "disabled"

        ### Creating a serverless index

        ```python
        import os
        from pinecone import Pinecone, ServerlessSpec

        client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        client.create_index(
            name="my_index",
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
            deletion_protection="enabled"
        )
        ```

        ### Creating a pod index

        ```python
        import os
        from pinecone import Pinecone, PodSpec

        client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        client.create_index(
            name="my_index",
            dimension=1536,
            metric="cosine",
            spec=PodSpec(
                environment="us-east1-gcp",
                pod_type="p1.x1"
            ),
            deletion_protection="enabled"
        )
        ```
        """

        api_instance = self.index_api

        if deletion_protection in ["enabled", "disabled"]:
            dp = DeletionProtection(deletion_protection)
        else:
            raise ValueError("deletion_protection must be either 'enabled' or 'disabled'")

        if isinstance(spec, dict):
            if "serverless" in spec:
                index_spec = IndexSpec(serverless=ServerlessSpecModel(**spec["serverless"]))
            elif "pod" in spec:
                args_dict = parse_non_empty_args(
                    [
                        ("environment", spec["pod"].get("environment")),
                        ("metadata_config", spec["pod"].get("metadata_config")),
                        ("replicas", spec["pod"].get("replicas")),
                        ("shards", spec["pod"].get("shards")),
                        ("pods", spec["pod"].get("pods")),
                        ("source_collection", spec["pod"].get("source_collection")),
                    ]
                )
                if args_dict.get("metadata_config"):
                    args_dict["metadata_config"] = PodSpecMetadataConfig(
                        indexed=args_dict["metadata_config"].get("indexed", None)
                    )
                index_spec = IndexSpec(pod=PodSpecModel(**args_dict))
            else:
                raise ValueError("spec must contain either 'serverless' or 'pod' key")
        elif isinstance(spec, ServerlessSpec):
            index_spec = IndexSpec(
                serverless=ServerlessSpecModel(cloud=spec.cloud, region=spec.region)
            )
        elif isinstance(spec, PodSpec):
            args_dict = parse_non_empty_args(
                [
                    ("replicas", spec.replicas),
                    ("shards", spec.shards),
                    ("pods", spec.pods),
                    ("source_collection", spec.source_collection),
                ]
            )
            if spec.metadata_config:
                args_dict["metadata_config"] = PodSpecMetadataConfig(
                    indexed=spec.metadata_config.get("indexed", None)
                )

            index_spec = IndexSpec(
                pod=PodSpecModel(environment=spec.environment, pod_type=spec.pod_type, **args_dict)
            )
        else:
            raise TypeError("spec must be of type dict, ServerlessSpec, or PodSpec")

        api_instance.create_index(
            create_index_request=CreateIndexRequest(
                name=name,
                dimension=dimension,
                metric=metric,
                spec=index_spec,
                deletion_protection=dp,
            )
        )

        def is_ready():
            status = self._get_status(name)
            ready = status["ready"]
            return ready

        if timeout == -1:
            return
        if timeout is None:
            while not is_ready():
                time.sleep(5)
        else:
            while (not is_ready()) and timeout >= 0:
                time.sleep(5)
                timeout -= 5
        if timeout and timeout < 0:
            raise (
                TimeoutError(
                    "Please call the describe_index API ({}) to confirm index status.".format(
                        "https://www.pinecone.io/docs/api/operation/describe_index/"
                    )
                )
            )

    def delete_index(self, name: str, timeout: Optional[int] = None):
        """Deletes a Pinecone index.

        Deleting an index is an irreversible operation. All data in the index will be lost.
        When you use this command, a request is sent to the Pinecone control plane to delete
        the index, but the termination is not synchronous because resources take a few moments to
        be released.

        You can check the status of the index by calling the `describe_index()` command.
        With repeated polling of the describe_index command, you will see the index transition to a
        `Terminating` state before eventually resulting in a 404 after it has been removed.

        :param name: the name of the index.
        :type name: str
        :param timeout: Number of seconds to poll status checking whether the index has been deleted. If None,
            wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait. Default: None
        :type timeout: int, optional
        """
        api_instance = self.index_api
        api_instance.delete_index(name)
        self.index_host_store.delete_host(self.config, name)

        def get_remaining():
            return name in self.list_indexes().names()

        if timeout == -1:
            return

        if timeout is None:
            while get_remaining():
                time.sleep(5)
        else:
            while get_remaining() and timeout >= 0:
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

    def list_indexes(self) -> IndexList:
        """Lists all indexes.

        The results include a description of all indexes in your project, including the
        index name, dimension, metric, status, and spec.

        :return: Returns an `IndexList` object, which is iterable and contains a
            list of `IndexModel` objects. It also has a convenience method `names()`
            which returns a list of index names.

        ```python
        from pinecone import Pinecone

        client = Pinecone()

        index_name = "my_index"
        if index_name not in client.list_indexes().names():
            print("Index does not exist, creating...")
            client.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
        ```

        You can also use the `list_indexes()` method to iterate over all indexes in your project
        and get other information besides just names.

        ```python
        from pinecone import Pinecone

        client = Pinecone()

        for index in client.list_indexes():
            print(index.name)
            print(index.dimension)
            print(index.metric)
            print(index.status)
            print(index.host)
            print(index.spec)
        ```

        """
        response = self.index_api.list_indexes()
        return IndexList(response)

    def describe_index(self, name: str):
        """Describes a Pinecone index.

        :param name: the name of the index to describe.
        :return: Returns an `IndexModel` object
        which gives access to properties such as the
        index name, dimension, metric, host url, status,
        and spec.

        ### Getting your index host url

        In a real production situation, you probably want to
        store the host url in an environment variable so you
        don't have to call describe_index and re-fetch it
        every time you want to use the index. But this example
        shows how to get the value from the API using describe_index.

        ```python
        from pinecone import Pinecone, Index

        client = Pinecone()

        description = client.describe_index("my_index")

        host = description.host
        print(f"Your index is hosted at {description.host}")

        index = client.Index(name="my_index", host=host)
        index.upsert(vectors=[...])
        ```
        """
        api_instance = self.index_api
        description = api_instance.describe_index(name)
        host = description.host
        self.index_host_store.set_host(self.config, name, host)

        return IndexModel(description)

    def has_index(self, name: str) -> bool:
        """Checks if a Pinecone index exists.

        :param name: The name of the index to check for existence.
        :return: Returns `True` if the index exists, `False` otherwise.

        ### Example Usage

        ```python
        import os
        from pinecone import Pinecone

        api_key = os.environ.get("PINECONE_API_KEY")
        pc = Pinecone(api_key=api_key)

        if pc.has_index("my_index_name"):
            print("The index exists")
        else:
            print("The index does not exist")
        ```
        """

        if name in self.list_indexes().names():
            return True
        else:
            return False

    def configure_index(
        self,
        name: str,
        replicas: Optional[int] = None,
        pod_type: Optional[str] = None,
        deletion_protection: Optional[Literal["enabled", "disabled"]] = None,
    ):
        """This method is used to scale configuration fields for your pod-based Pinecone index.

        :param: name: the name of the Index
        :param: replicas: the desired number of replicas, lowest value is 0.
        :param: pod_type: the new pod_type for the index. To learn more about the
            available pod types, please see [Understanding Indexes](https://docs.pinecone.io/docs/indexes)


        ```python
        from pinecone import Pinecone

        client = Pinecone()

        # Make a configuration change
        client.configure_index(name="my_index", replicas=4)

        # Call describe_index to see the index status as the
        # change is applied.
        client.describe_index("my_index")
        ```

        """
        api_instance = self.index_api

        if deletion_protection is None:
            description = self.describe_index(name=name)
            dp = DeletionProtection(description.deletion_protection)
        elif deletion_protection in ["enabled", "disabled"]:
            dp = DeletionProtection(deletion_protection)
        else:
            raise ValueError("deletion_protection must be either 'enabled' or 'disabled'")

        pod_config_args: Dict[str, Any] = {}
        if pod_type:
            pod_config_args.update(pod_type=pod_type)
        if replicas:
            pod_config_args.update(replicas=replicas)

        if pod_config_args != {}:
            spec = ConfigureIndexRequestSpec(pod=ConfigureIndexRequestSpecPod(**pod_config_args))
            req = ConfigureIndexRequest(deletion_protection=dp, spec=spec)
        else:
            req = ConfigureIndexRequest(deletion_protection=dp)

        api_instance.configure_index(name, configure_index_request=req)

    def create_collection(self, name: str, source: str):
        """Create a collection from a pod-based index

        :param name: Name of the collection
        :param source: Name of the source index
        """
        api_instance = self.index_api
        api_instance.create_collection(
            create_collection_request=CreateCollectionRequest(name=name, source=source)
        )

    def list_collections(self) -> CollectionList:
        """List all collections

        ```python
        from pinecone import Pinecone

        client = Pinecone()

        for collection in client.list_collections():
            print(collection.name)
            print(collection.source)

        # You can also iterate specifically over the collection
        # names with the .names() helper.
        collection_name="my_collection"
        for collection_name in client.list_collections().names():
            print(collection_name)
        ```
        """
        api_instance = self.index_api
        response = api_instance.list_collections()
        return CollectionList(response)

    def delete_collection(self, name: str):
        """Deletes a collection.

        :param: name: The name of the collection

        Deleting a collection is an irreversible operation. All data
        in the collection will be lost.

        This method tells Pinecone you would like to delete a collection,
        but it takes a few moments to complete the operation. Use the
        `describe_collection()` method to confirm that the collection
        has been deleted.
        """
        api_instance = self.index_api
        api_instance.delete_collection(name)

    def describe_collection(self, name: str):
        """Describes a collection.
        :param: The name of the collection
        :return: Description of the collection

        ```python
        from pinecone import Pinecone

        client = Pinecone()

        description = client.describe_collection("my_collection")
        print(description.name)
        print(description.source)
        print(description.status)
        print(description.size)
        ```
        """
        api_instance = self.index_api
        return api_instance.describe_collection(name).to_dict()

    def _get_status(self, name: str):
        api_instance = self.index_api
        response = api_instance.describe_index(name)
        return response["status"]

    @staticmethod
    def from_texts(*args, **kwargs):
        raise AttributeError(_build_langchain_attribute_error_message("from_texts"))

    @staticmethod
    def from_documents(*args, **kwargs):
        raise AttributeError(_build_langchain_attribute_error_message("from_documents"))

    def Index(self, name: str = "", host: str = "", **kwargs):
        """
        Target an index for data operations.

        ### Target an index by host url

        In production situations, you want to uspert or query your data as quickly
        as possible. If you know in advance the host url of your index, you can
        eliminate a round trip to the Pinecone control plane by specifying the
        host of the index.

        ```python
        import os
        from pinecone import Pinecone

        api_key = os.environ.get("PINECONE_API_KEY")
        index_host = os.environ.get("PINECONE_INDEX_HOST")

        pc = Pinecone(api_key=api_key)
        index = pc.Index(host=index_host)

        # Now you're ready to perform data operations
        index.query(vector=[...], top_k=10)
        ```

        To find your host url, you can use the Pinecone control plane to describe
        the index. The host url is returned in the response. Or, alternatively, the
        host is displayed in the Pinecone web console.

        ```python
        import os
        from pinecone import Pinecone

        pc = Pinecone(
            api_key=os.environ.get("PINECONE_API_KEY")
        )

        host = pc.describe_index('index-name').host
        ```

        ### Target an index by name (not recommended for production)

        For more casual usage, such as when you are playing and exploring with Pinecone
        in a notebook setting, you can also target an index by name. If you use this
        approach, the client may need to perform an extra call to the Pinecone control
        plane to get the host url on your behalf to get the index host.

        The client will cache the index host for future use whenever it is seen, so you
        will only incur the overhead of only one call. But this approach is not
        recommended for production usage.

        ```python
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
        ```
        """
        if name == "" and host == "":
            raise ValueError("Either name or host must be specified")

        pt = kwargs.pop("pool_threads", None) or self.pool_threads
        api_key = self.config.api_key
        openapi_config = self.openapi_config

        if host != "":
            # Use host url if it is provided
            index_host = normalize_host(host)
        else:
            # Otherwise, get host url from describe_index using the index name
            index_host = self.index_host_store.get_host(self.index_api, self.config, name)

        return Index(
            host=index_host,
            api_key=api_key,
            pool_threads=pt,
            openapi_config=openapi_config,
            source_tag=self.config.source_tag,
            **kwargs,
        )
