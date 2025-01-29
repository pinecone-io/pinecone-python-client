from abc import ABC, abstractmethod

from typing import Optional, Dict, Union


from pinecone.config import Config

from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi


from pinecone.models import (
    ServerlessSpec,
    PodSpec,
    IndexList,
    CollectionList,
    IndexModel,
    IndexEmbed,
)
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
from .types import CreateIndexForModelEmbedTypedDict


class PineconeDBControlInterface(ABC):
    @abstractmethod
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

    pass

    @abstractmethod
    def create_index(
        self,
        name: str,
        spec: Union[Dict, ServerlessSpec, PodSpec],
        dimension: Optional[int],
        metric: Optional[Union[Metric, str]] = Metric.COSINE,
        timeout: Optional[int] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = DeletionProtection.DISABLED,
        vector_type: Optional[Union[VectorType, str]] = VectorType.DENSE,
    ):
        """Creates a Pinecone index.

        :param name: The name of the index to create. Must be unique within your project and
            cannot be changed once created. Allowed characters are lowercase letters, numbers,
            and hyphens and the name may not begin or end with hyphens. Maximum length is 45 characters.
        :type name: str
        :param metric: Type of metric used in the vector index when querying, one of `{"cosine", "dotproduct", "euclidean"}`. Defaults to `"cosine"`.
            Defaults to `"cosine"`.
        :type metric: str, optional
        :param spec: A dictionary containing configurations describing how the index should be deployed. For serverless indexes,
            specify region and cloud. For pod indexes, specify replicas, shards, pods, pod_type, metadata_config, and source_collection.
        :type spec: Dict
        :param dimension: The dimension of vectors that will be inserted in the index. This should
            match the dimension of the embeddings you will be inserting. For example, if you are using
            OpenAI's CLIP model, you should use `dimension=1536`. Dimension is a required field when
            creating an index with vector_type="dense".
        :type dimension: int
        :type timeout: int, optional
        :param timeout: Specify the number of seconds to wait until index gets ready. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait. Default: None
        :param deletion_protection: If enabled, the index cannot be deleted. If disabled, the index can be deleted. Default: "disabled"
        :param vector_type: The type of vectors to be stored in the index. One of `{"dense", "sparse"}`. Default: "dense"
        :type vector_type: str, optional

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
        pass

    @abstractmethod
    def create_index_for_model(
        self,
        name: str,
        cloud: Union[CloudProvider, str],
        region: Union[AwsRegion, GcpRegion, AzureRegion, str],
        embed: Union[IndexEmbed, CreateIndexForModelEmbedTypedDict],
        tags: Optional[Dict[str, str]] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = DeletionProtection.DISABLED,
        timeout: Optional[int] = None,
    ) -> IndexModel:
        """
        Create an index for a model.

        This operation creates a serverless index for a model. The index is used to store embeddings generated by the model. The index can be used to search and retrieve embeddings.

        :param name: The name of the index to create. Must be unique within your project and
            cannot be changed once created. Allowed characters are lowercase letters, numbers,
            and hyphens and the name may not begin or end with hyphens. Maximum length is 45 characters.
        :type name: str
        :param cloud: The cloud provider to use for the index.
        :type cloud: str
        :param region: The region to use for the index.
        :type region: str
        :param embed: The embedding configuration for the index.
        :type embed: Union[Dict, IndexEmbed]
        :param tags: A dictionary of tags to associate with the index.
        :type tags: Optional[Dict[str, str]]
        :param deletion_protection: If enabled, the index cannot be deleted. If disabled, the index can be deleted. Default: "disabled"
        :type deletion_protection: Optional[Literal["enabled", "disabled"]]
        :type timeout: Optional[int]
        :param timeout: Specify the number of seconds to wait until index gets ready. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait. Default: None
        :return: The index that was created.
        :rtype: IndexModel
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def describe_index(self, name: str) -> IndexModel:
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def configure_index(
        self,
        name: str,
        replicas: Optional[int] = None,
        pod_type: Optional[Union[PodType, str]] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """This method is used to scale configuration fields for your pod-based Pinecone index.

        :param: name: the name of the Index
        :param: replicas: the desired number of replicas, lowest value is 0.
        :param: pod_type: the new pod_type for the index. To learn more about the
            available pod types, please see [Understanding Indexes](https://docs.pinecone.io/docs/indexes)
        :param: deletion_protection: If set to 'enabled', the index cannot be deleted. If 'disabled', the index can be deleted.
        :param: tags: A dictionary of tags to apply to the index. Tags are key-value pairs that can be used to organize and manage indexes. To remove a tag, set the value to "". Tags passed to configure_index will be merged with existing tags and any with the value empty string will be removed.

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
        pass

    @abstractmethod
    def create_collection(self, name: str, source: str):
        """Create a collection from a pod-based index

        :param name: Name of the collection
        :param source: Name of the source index
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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

        Arguments:
            name: The name of the index to target. If you specify the name of the index, the client will
                fetch the host url from the Pinecone control plane.
            host: The host url of the index to target. If you specify the host url, the client will use
                the host url directly without making any additional calls to the control plane.
            pool_threads: The number of threads to use when making parallel requests by calling index methods with optional kwarg async_req=True, or using methods that make use of parallelism automatically such as query_namespaces(). Default: 1
            connection_pool_maxsize: The maximum number of connections to keep in the connection pool. Default: 5 * multiprocessing.cpu_count()
        """
        pass
