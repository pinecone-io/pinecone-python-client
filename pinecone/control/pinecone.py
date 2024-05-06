import time
import httpx
import warnings
from typing import Optional, Dict

from ..utils import get_user_agent
from .index_host_store import IndexHostStore

from pinecone.config import PineconeConfig, Config, ConfigBuilder

from pinecone.core.client.api.manage_indexes_api import ManageIndexesApi
from pinecone.utils import normalize_host, setup_openapi_client
from pinecone.models import (
    CollectionList
)
from pinecone.core.client.models import (
    CreateCollectionRequest,
)

from pinecone.data import Index

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
                **kwargs
            )

        self.http_client = httpx.Client(
            headers={
                'User-Agent': get_user_agent(config),
            },
            base_url=self.config.host
        )

        if kwargs.get("openapi_config", None):
            warnings.warn("Passing openapi_config is deprecated and will be removed in a future release. Please pass settings such as proxy_url, proxy_headers, ssl_ca_certs, and ssl_verify directly to the Pinecone constructor as keyword arguments. See the README at https://github.com/pinecone-io/pinecone-python-client for examples.", DeprecationWarning)

        self.openapi_config = ConfigBuilder.build_openapi_config(self.config, **kwargs)
        self.pool_threads = pool_threads

        if index_api:
            self.index_api = index_api
        else:
            self.index_api = setup_openapi_client(ManageIndexesApi, self.config, self.openapi_config, pool_threads)

        self.index_host_store = IndexHostStore()
        """ @private """


    def create_collection(self, name: str, source: str):
        """Create a collection from a pod-based index

        :param name: Name of the collection
        :param source: Name of the source index
        """
        api_instance = self.index_api
        api_instance.create_collection(create_collection_request=CreateCollectionRequest(name=name, source=source))

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
        print(description.)
        ```
        """
        api_instance = self.index_api
        return api_instance.describe_collection(name).to_dict()

    def _get_status(self, name: str):
        api_instance = self.index_api
        response = api_instance.describe_index(name)
        return response["status"]


    def Index(self, name: str = '', host: str = '', **kwargs):
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
            name='my-index',
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )
        index = pc.Index('my-index')

        # Now you're ready to perform data operations
        index.query(vector=[...], top_k=10)
        ```
        """
        if name == '' and host == '':
            raise ValueError("Either name or host must be specified")
        
        pt = kwargs.pop('pool_threads', None) or self.pool_threads
        api_key = self.config.api_key
        openapi_config = self.openapi_config

        if host != '':
            # Use host url if it is provided
            index_host=normalize_host(host)
        else:
            # Otherwise, get host url from describe_index using the index name
            index_host = self.index_host_store.get_host(self.index_api, self.config, name)

        return Index(
            host=index_host,
            api_key=api_key,
            pool_threads=pt,
            openapi_config=openapi_config,
            source_tag=self.config.source_tag,
            **kwargs
        )