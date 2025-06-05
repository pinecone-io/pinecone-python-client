# FAQ

## How does connection pooling work in the Pinecone SDK?

Before any data can be sent or received from Pinecone, your application must first establish a TCP connection with our API gateway. Establishing a TCP connection is a costly operation from a performance standpoint and so we use connection pooling to cache and reuse these connections across many different requests.

Every instance of the `Pinecone` or `Index` client (and variant classes such as `PineconeAsyncio`, `AsyncioIndex`, `PineconeGRPC`, and `IndexGRPC`) maintains its own connection pool.

This means that in order to get the benefits of connection pooling in your application, you should cache your client instance instead of recreating it for every operation. If you create a new client instance for every call your application makes for Pinecone, you will see significantly worse performance due to the TCP connection churn.

## When are TCP connections established?

When using the Pinecone python client, no connections are made when first instantiating the client by writing a statement like `pc = Pinecone()`. The only thing that happens during the client instantiation is taking in configuration parameters such as `api_key`, `host`, and setting up objects needed in memory in preparation for subsequent interactions.

A TCP connection is not established until you attempt to invoke a command resulting in network operation such as `upsert`, `query`, etc. Connections created during these operations are cached in a connection pool for reuse by subsequent requests.

The specific details on how connection pooling is done varies across the different client versions.

- The `Pinecone` client relies on urllib3's [`PoolManager`](https://urllib3.readthedocs.io/en/stable/reference/urllib3.poolmanager.html#urllib3.PoolManager).
- `PineconeAsyncio` relies on the automatic connection pooling behavior of [`aiohttp`](https://docs.aiohttp.org/en/stable/client_advanced.html)

## How can I verify my API key, proxy, or other settings are valid?

### Connecting to `api.pinecone.io`

To verify you can reach `api.pinecone.io` you should instantiate the client and then call `list_indexes()`. This command is convenient because it will make a network call to Pinecone but doesn't require any arguments or prerequisite steps to get a successful response. This will return a response if your API key is valid even if you have not created any indexes in your project.

```python
import os
from pinecone import Pinecone

pc = Pinecone(
    api_key=os.environ['PINECONE_API_KEY']
)
pc.list_indexes()
```

### Connecting to your index

Each Pinecone index has a unique host. To check whether you can connect to your index, you can run `describe_index_stats()` or `fetch`. For fetch, you must pass an id but if the id does not exist (for example, if your index is empty because you just created it, or you don't know any specific ids) you will get a response that includes an empty result set. This should be fine for the purpose of checking whether you're able to complete network calls successfully.

You can look up your index host through the [Pinecone web console](https://api.pinecone.io) or by calling in to `api.pinecone.io` with `pc.describe_index(name='index-name').host`. Using the SDK to get this value is convenient when testing interactively in a small script or notebook setting, but in production you should lookup the host and store it in an environment variable to remove an unnecessary runtime dependency on `api.pinecone.io`.

```python
from pinecone import Pinecone

pc = Pinecone(
    api_key=os.environ['PINECONE_API_KEY'],
    proxy_url='https://your-proxy.com',
    ssl_ca_certs='path/to/cert-bundle.pem'
)

idx = pc.Index(host="example-index-dojoi3u.svc.eu-west1-gcp.pinecone.io")
idx.fetch(ids=['testid'])
# If you reach this point with no exceptions raised, you should be good.
```

### Does the Pinecone Python SDK use HTTP/2?

The answer to this varies by client variant.

- `Pinecone` relies on `urllib3` for network calls which only supports `HTTP/1.1`
- `PineconeAsyncio` (requires installing extra dependencies with `pinecone[asyncio]`) relies on `aiohttp` for network calls which only supports `HTTP/1.1`
- `PineconeGRPC` (requires installing extra dependencies with `pinecone[grpc]`)
relies on `grpcio` for network calls and does use `HTTP/2`.

Over time we anticipate reevaluating these technology choices and adopting HTTP/2 to unlock further performance improvements.
