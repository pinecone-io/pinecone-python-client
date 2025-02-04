
# Advanced topics

## Proxy configuration

If your network setup requires you to interact with Pinecone via a proxy, you will need
to pass additional configuration using optional keyword parameters. These optional parameters are forwarded to `urllib3`, which is the underlying library currently used by the Pinecone SDK to make HTTP requests. You may find it helpful to refer to the [urllib3 documentation on working with proxies](https://urllib3.readthedocs.io/en/stable/advanced-usage.html#http-and-https-proxies) while troubleshooting these settings.

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

By default the Pinecone Python SDK will perform SSL certificate verification
using the CA bundle maintained by Mozilla in the [certifi](https://pypi.org/project/certifi/) package.

If your proxy server is using a self-signed certificate, you will need to pass the path to the certificate in PEM format using the `ssl_ca_certs` parameter.

```python
from pinecone import Pinecone
import urllib3 import make_headers

pc = Pinecone(
    api_key="YOUR_API_KEY",
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

### Working with GRPC (for improved performance)

If you've followed instructions above to install with optional `grpc` extras, you can unlock some performance improvements by working with an alternative version of the SDK imported from the `pinecone.grpc` subpackage.

```python
import os
from pinecone.grpc import PineconeGRPC

pc = PineconeGRPC(api_key=os.environ.get('PINECONE_API_KEY'))

# From here on, everything is identical to the REST-based SDK.
index = pc.Index(host='my-index-8833ca1.svc.us-east1-gcp.pinecone.io')

index.upsert(vectors=[])
index.query(vector=[...], top_key=10)
```
