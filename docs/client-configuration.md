### Initializing the client

Before you can use the Pinecone SDK, you must sign up for an account and find your API key in the Pinecone console dashboard at [https://app.pinecone.io](https://app.pinecone.io).

#### Configuration with environment variables

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

#### Using configuration keyword params

If you prefer to pass configuration in code, for example if you have a complex application that needs to interact with multiple different Pinecone projects, the constructor accepts a keyword argument for `api_key`.

If you pass configuration in this way, you can have full control over what name to use for the environment variable, sidestepping any issues that would result
from two different client instances both needing to read the same `PINECONE_API_KEY` variable that the client implicitly checks for.

Configuration passed with keyword arguments takes precedence over environment variables.

```python
import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ.get('CUSTOM_VAR'))
```
