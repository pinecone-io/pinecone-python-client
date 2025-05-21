# Debugging the Pinecone SDK

## Enabling debug logging for REST, asyncio

You can turn on detailed debug logging if needed, but it's a little bit challenging because it's not currently exposed to the user in a nice way. You have to reach into the internals a bit after the client is instantiated to see everything.

> [!WARNING]
> Be careful with this output as it will leak headers with secrets, including the `Api-Key` header. I manually redacted that value from this example below.

If I defined a script like this in a file `scripts/repro.py`:

```python
import dotenv
import logging
from pinecone import Pinecone

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)

pc = Pinecone()
pc._openapi_config.debug = True
pc.describe_index('jen')
```

Running it with `poetry run python3 scripts/repro.py` would give output like

```
DEBUG    | pinecone.openapi_support.rest_urllib3:125 | Calling urllib3 request()
send: b'GET /indexes/jen HTTP/1.1\r\nHost: api.pinecone.io\r\nAccept-Encoding: identity\r\nAccept: application/json\r\nUser-Agent: python-client-6.0.2\r\nx-environment: preprod-aws-0\r\nX-Pinecone-API-Version: 2025-04\r\nApi-Key: REDACTEDX\r\n\r\n'
reply: 'HTTP/1.1 200 OK\r\n'
header: content-type: application/json
header: access-control-allow-origin: *
header: vary: origin,access-control-request-method,access-control-request-headers
header: access-control-expose-headers: *
header: x-pinecone-api-version: 2025-04
header: x-cloud-trace-context: ac668597d0413fd780f6d9536f80195b
header: date: Wed, 21 May 2025 16:48:08 GMT
header: server: Google Frontend
header: Content-Length: 263
header: Via: 1.1 google
header: Alt-Svc: h3=":443"; ma=2592000,h3-29=":443"; ma=2592000
DEBUG    | urllib3.connectionpool:546 | https://api.pinecone.io:443 "GET /indexes/jen HTTP/11" 200 0
DEBUG    | pinecone.openapi_support.rest_urllib3:265 | response body: b'{"name":"jen","vector_type":"dense","metric":"cosine","dimension":2,"status":{"ready":true,"state":"Ready"},"host":"jen-dojoi3u.svc.preprod-aws-0.pinecone.io","spec":{"serverless":{"region":"us-east-1","cloud":"aws"}},"deletion_protection":"disabled","tags":null}'
DEBUG    | pinecone.openapi_support.rest_utils:34 | response status: 200
{
    "name": "jen",
    "metric": "cosine",
    "host": "jen-dojoi3u.svc.preprod-aws-0.pinecone.io",
    "spec": {
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1"
        }
    },
    "status": {
        "ready": true,
        "state": "Ready"
    },
    "vector_type": "dense",
    "dimension": 2,
    "deletion_protection": "disabled",
    "tags": null
}
```

## Enabling debug logging for GRPC

Debug output for GRPC is controlled with [environment variables](https://github.com/grpc/grpc/blob/master/doc/environment_variables.md). Set `GRPC_TRACE='all'`.

## Using breakpoints

Python has a built-in debugger called [pdb](https://docs.python.org/3/library/pdb.html).

Basic usage involves inserting a call to `breakpoint()` into your program. This will halt when reached during execution and drop you into a REPL that allows you to explore the local variables at that point of the execution.

Once you're in the pdb session, you can inspect variables, advance line by line using `next`, or resume execution using `continue`. This can be a really useful technique for getting to the bottom of a problem when working on a complex integration test or doing manual testing in the repl.

A useful spot to insert the `breakpoint()` invocation is inside the `request` method of the `Urllib3RestClient` or `AiohttpRestClient` classes. After making an edit to insert a `breakpoint()` invocation in my request method, I can inspect the request params like this:

```sh
poetry run repl

    Welcome to the custom Python REPL!
    Your initialization steps have been completed.

    Two Pinecone objects are available:
    - pc: Interact with the one-offs project
    - pcci: Interact with the pinecone-python-client project (CI testing)

    You can use the following functions to clean up the environment:
    - delete_all_indexes(pc)
    - delete_all_collections(pc)
    - delete_all_backups(pc)
    - cleanup_all(pc)

>>> pc.describe_index('jen')
> /Users/jhamon/workspace/pinecone-python-client/pinecone/openapi_support/rest_urllib3.py(127)request()
-> method = method.upper()
(Pdb) method
'GET'
(Pdb) url
'https://api.pinecone.io/indexes/jen'
(Pdb) next
> /Users/jhamon/workspace/pinecone-python-client/pinecone/openapi_support/rest_urllib3.py(128)request()
-> assert method in ["GET", "HEAD", "DELETE", "POST", "PUT", "PATCH", "OPTIONS"]
(Pdb) next
> /Users/jhamon/workspace/pinecone-python-client/pinecone/openapi_support/rest_urllib3.py(130)request()
-> if os.environ.get("PINECONE_DEBUG_CURL"):
(Pdb) next
> /Users/jhamon/workspace/pinecone-python-client/pinecone/openapi_support/rest_urllib3.py(158)request()
-> if post_params and body:
(Pdb) continue
{
    "name": "jen",
    "metric": "cosine",
    "host": "jen-dojoi3u.svc.preprod-aws-0.pinecone.io",
    "spec": {
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1"
        }
    },
    "status": {
        "ready": true,
        "state": "Ready"
    },
    "vector_type": "dense",
    "dimension": 2,
    "deletion_protection": "disabled",
    "tags": null
}
```

## Reporting errors to backend teams

Sometimes errors are caused by unexpected behavior in the underlying API. Once you have confirmed this is the case, you need to convey that information to the appropriate backend teams in a concise way that removes all doubt that the SDK is to blame.

You can set the environment variable `PINECONE_DEBUG_CURL='true'` to see some printed output approximating what the REST client does translated into curl calls. This is useful for reporting API problems in a way that is copy/pasteable to backend teams for easy reproducibility without all the hassle of setting up a python notebook to repro. Be aware that this output will leak your API key.

> [!WARNING]
> Output from `PINECONE_DEBUG_CURL='true'` will include your secret API key. Do not use it in production environments and be careful when sharing the output.
