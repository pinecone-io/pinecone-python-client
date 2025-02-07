# Initializing the client


## Configuration with environment variables

The Pinecone Python SDK will read the following environment variables:

- `PINECONE_API_KEY` can be set in place of passing an `api_key` keyword argument to the `Pinecone`, `PineconeAsyncio`, `PineconeGRPC`, or constructors.
- `PINECONE_DEBUG_CURL` can be used to enable some additional debug output for `Pinecone`. When troubleshooting it can be very useful to run curl
    commands against the control plane API to see exactly what data is being sent
    and received without all the abstractions and transformations applied by the Python
    SDK. If you set this environment variable to `true`, the Pinecone client will use
    request parameters to print out an equivalent curl command that you can run yourself
    or share with Pinecone support. **Be very careful with this option, as it will print out
    your API key** which forms part of a required authentication header. The main use of
    is to help evaluate whether a problem you are experiencing is due to the API's behavior
    or the behavior of the SDK itself.

```python
from pinecone import Pinecone

pc = Pinecone() # This reads the PINECONE_API_KEY env var
```

## Using configuration keyword params

If you prefer to pass configuration in code, for example if you have a complex application that needs to interact with multiple different Pinecone projects, the constructor accepts a keyword argument for `api_key`.

If you pass configuration in this way, you can have full control over what name to use for the environment variable, sidestepping any issues that would result
from two different client instances both needing to read the same `PINECONE_API_KEY` variable that the client implicitly checks for.

Configuration passed with keyword arguments takes precedence over environment variables.

```python
import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ.get('CUSTOM_VAR'))
```

## Proxy configuration

If your network setup requires you to interact with Pinecone via a proxy, you will need to pass additional configuration using optional keyword parameters. These optional parameters are forwarded to `urllib3` (or `aiohttp` when using `PineconeAsyncio`), which is the underlying library currently used by the Pinecone SDK to make HTTP requests. You may find it helpful to refer to the [urllib3 documentation on working with proxies](https://urllib3.readthedocs.io/en/stable/advanced-usage.html#http-and-https-proxies) while troubleshooting these settings.

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
