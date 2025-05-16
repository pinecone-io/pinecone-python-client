> [!NOTE]
> The official SDK package was renamed from `pinecone-client` to `pinecone` beginning in version 5.1.0.
> Please remove `pinecone-client` from your project dependencies and add `pinecone` instead to get
> the latest updates.

# Upgrading from `6.x` to `7.x`

There are no intentional breaking changes when moving from v6 to v7 of the SDK. The major version bump reflects the move from calling the `2025-01` to the `2025-04` version of the underlying API.

Some internals of the client have been reorganized or moved, but we've made an effort to alias everything and show warning messages when appropriate. If you experience any unexpected breaking changes that cause you friction while upgrading, let us know and we'll try to smooth it out.

## Useful additions in `7.x`

New Features:
- [Pinecone Assistant](https://www.pinecone.io/product/assistant/): The assistant plugin is now bundled by default. You can simply start using it without installing anything additional.
- [Inference API](https://docs.pinecone.io/guides/get-started/overview#inference): List/view models from the model gallery via API
- [Backups](https://docs.pinecone.io/guides/manage-data/backups-overview):
  - Create backup from serverless index
  - Create serverless index from backup
  - List/view backups
  - List/view backup restore jobs
- [Bring Your Own Cloud (BYOC)](https://docs.pinecone.io/guides/production/bring-your-own-cloud):
  - Create, list, describe, and delete BYOC indexes

Other improvements:
- ~70% faster client instantiation time thanks to extensive refactoring to implement lazy loading. This means your app won't waste time loading code for features you're not using.
- Retries with exponential backoff are now enabled by default for REST calls (implemented for both urllib3 and aiohttp).
- We're following [PEP 561](https://typing.python.org/en/latest/spec/distributing.html#packaging-typed-libraries) and adding a `py.typed` marker file to indicate inline type information is present in the package. We're still working toward reaching full coverage with our type hints, but including this file allows some tools to find the inline definitions we have already implemented.


### Backups for Serverless Indexes

You can create backups from your serverless indexes and use these backups to create new indexes. Some fields such as `record_count` are initially empty but will be populated by the time a backup is ready for use.

```python
from pinecone import Pinecone

pc = Pinecone()

index_name = 'example-index'
if not pc.has_index(name=index_name):
    raise Exception('An index must exist before backing it up')

backup = pc.create_backup(
    index_name=index_name,
    backup_name='example-backup',
    description='testing out backups'
)
# {
#     "backup_id": "4698a618-7e56-4a44-93bc-fc8f1371aa36",
#     "source_index_name": "example-index",
#     "source_index_id": "ec6fd44c-ab45-4873-97f3-f6b44b67e9bc",
#     "status": "Initializing",
#     "cloud": "aws",
#     "region": "us-east-1",
#     "tags": {},
#     "name": "example-backup",
#     "description": "testing out backups",
#     "dimension": null,
#     "record_count": null,
#     "namespace_count": null,
#     "size_bytes": null,
#     "created_at": "2025-05-16T18:44:28.480671533Z"
# }
```

Check the status of a backup

```python
from pinecone import Pinecone

pc = Pinecone()

pc.describe_backup(backup_id='4698a618-7e56-4a44-93bc-fc8f1371aa36')
# {
#     "backup_id": "4698a618-7e56-4a44-93bc-fc8f1371aa36",
#     "source_index_name": "example-index",
#     "source_index_id": "ec6fd44c-ab45-4873-97f3-f6b44b67e9bc",
#     "status": "Ready",
#     "cloud": "aws",
#     "region": "us-east-1",
#     "tags": {},
#     "name": "example-backup",
#     "description": "testing out backups",
#     "dimension": 768,
#     "record_count": 1000,
#     "namespace_count": 1,
#     "size_bytes": 289656,
#     "created_at": "2025-05-16T18:44:28.480691Z"
# }
```

You can use `list_backups` to see all of your backups and their current status. If you have a large number of backups, results will be paginated. You can control the pagination with optional parameters for `limit` and `pagination_token`.

```python

from pinecone import Pinecone

pc = Pinecone()

# All backups
pc.list_backups()

# Only backups associated with a particular index
pc.list_backups(index_name='my-index')
```

To create an index from a backup, use `create_index_from_backup`.

```python
from pinecone import Pinecone

pc = Pinecone()

pc.create_index_from_backup(
    name='index-from-backup',
    backup_id='4698a618-7e56-4a44-93bc-fc8f1371aa36',
    deletion_protection = "disabled",
    tags={'env': 'testing'},
)
```

Under the hood, a restore job is created to handle taking data from your backup and loading it into the newly created serverless index. You can check status of pending restore jobs with `pc.list_restore_jobs()` or `pc.describe_restore_job()`

### Explore and discover models available in our Inference API

You can now fetch a dynamic list of models supported by the Inference API.

```python
from pinecone import Pinecone

pc = Pinecone()

# List all models
models = pc.inference.list_models()

# List models, with model type filtering
models = pc.inference.list_models(type="embed")
models = pc.inference.list_models(type="rerank")

# List models, with vector type filtering
models = pc.inference.list_models(vector_type="dense")
models = pc.inference.list_models(vector_type="sparse")

# List models, with both type and vector type filtering
models = pc.inference.list_models(type="rerank", vector_type="dense")
```

Or, if you know the name of a model, you can get just those details

```
pc.inference.get_model(model_name='pinecone-rerank-v0')
# {
#     "model": "pinecone-rerank-v0",
#     "short_description": "A state of the art reranking model that out-performs competitors on widely accepted benchmarks. It can handle chunks up to 512 tokens (1-2 paragraphs)",
#     "type": "rerank",
#     "supported_parameters": [
#         {
#             "parameter": "truncate",
#             "type": "one_of",
#             "value_type": "string",
#             "required": false,
#             "default": "END",
#             "allowed_values": [
#                 "END",
#                 "NONE"
#             ]
#         }
#     ],
#     "modality": "text",
#     "max_sequence_length": 512,
#     "max_batch_size": 100,
#     "provider_name": "Pinecone",
#     "supported_metrics": []
# }
```

### Client support for BYOC (Bring Your Own Cloud)

For customers using our [BYOC offering](https://docs.pinecone.io/guides/production/bring-your-own-cloud), you can now create indexes and list/describe indexes you have created in your cloud.

```python
from pinecone import Pinecone, ByocSpec

pc = Pinecone()

pc.create_index(
    name='example-byoc-index',
    dimension=768,
    metric='cosine',
    spec=ByocSpec(environment='my-private-env'),
    tags={
        'env': 'testing'
    },
    deletion_protection='enabled'
)
```

# Upgrading from `5.x` to `6.x`

## Breaking changes in 6.x
- Dropped support for Python 3.8, which has now reached [official end of life](https://devguide.python.org/versions/). We added support for Python 3.13.
- Removed the explicit dependency on [`tqdm`](https://github.com/tqdm/tqdm) which is used to provide a nice progress bar when upserting lots of data into Pinecone. If `tqdm` is available in the environment the Pinecone SDK will detect and use it but we will no longer require `tqdm` to be installed in order to run the SDK. Popular notebook platforms such as [Jupyter](https://jupyter.org/) and [Google Colab](https://colab.google/) already include `tqdm` in the environment by default so for many users this will not require any changes, but if you are running small scripts in other environments and want to continue seeing the progress bars you will need to separately install the `tqdm` package.
- Removed some previously deprecated and rarely used keyword arguments (`config`, `openapi_config`, and `index_api`) to instead prefer dedicated keyword arguments for individual settings such as `api_key`, `proxy_url`, etc. These keyword arguments were primarily aimed at facilitating testing but were never documented for the end-user so we expect few people to be impacted by the change. Having multiple ways of passing in the same configuration values was adding significant amounts of complexity to argument validation, testing, and documentation that wasn't really being repaid by significant ease of use, so we've removed those options.

## Useful additions in 6.x:

## Compatibility with `asyncio`

The v6 Python SDK introduces a new client variants, `PineconeAsyncio` and `IndexAsyncio`, which provide `async` methods for use with [asyncio](https://docs.python.org/3/library/asyncio.html). This should unblock those who wish to use Pinecone with modern async web frameworks such as [FastAPI](https://fastapi.tiangolo.com/), [Quart](https://quart.palletsprojects.com/en/latest/), [Sanic](https://sanic.dev/), etc.

Those trying to onboard to Pinecone and upsert large amounts of data should significantly benefit from the efficiency of running many upserts in parallel.

Calls to `api.pinecone.io` for creating, configuring, or deleting indexes are made with `PineconeAsyncio`.

```python
import asyncio

from pinecone import (
    PineconeAsyncio,
    IndexEmbed,
    CloudProvider,
    AwsRegion,
    EmbedModel
)

async def main():
    async with PineconeAsyncio() as pc:
        if not await pc.has_index(index_name):
            desc = await pc.create_index_for_model(
                name="book-search",
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1,
                embed=IndexEmbed(
                    model=EmbedModel.Multilingual_E5_Large,
                    metric="cosine",
                    field_map={
                        "text": "description",
                    },
                )
            )

asyncio.run(main())
```

Interactions with a deployed index are done via `IndexAsyncio`:

```python
import asyncio

from pinecone import Pinecone

async def main():
    host="book-search-dojoi3u.svc.aped-4627-b74a.pinecone.io"
    async with Pinecone().IndexAsyncio(host=host) as idx:
        await idx.upsert_records(
            namespace="",
            records=[
                {
                    "id": "1",
                    "title": "The Great Gatsby",
                    "author": "F. Scott Fitzgerald",
                    "description": "The story of the mysteriously wealthy Jay Gatsby and his love for the beautiful Daisy Buchanan.",
                    "year": 1925,
                },
                {
                    "id": "2",
                    "title": "To Kill a Mockingbird",
                    "author": "Harper Lee",
                    "description": "A young girl comes of age in the segregated American South and witnesses her father's courageous defense of an innocent black man.",
                    "year": 1960,
                },
                {
                    "id": "3",
                    "title": "1984",
                    "author": "George Orwell",
                    "description": "In a dystopian future, a totalitarian regime exercises absolute control through pervasive surveillance and propaganda.",
                    "year": 1949,
                },
            ]
        )


asyncio.run(main())
```


### Sparse indexes

These are created using the same methods as before but using different configuration options. For sparse indexes, you must omit `dimension`, `metric="dotproduct`, and `vector_type="sparse"`.

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    Metric,
    VectorType
)

pc = Pinecone()
pc.create_index(
    name='sparse-index',
    metric=Metric.DOTPRODUCT,
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_WEST_2
    ),
    vector_type=VectorType.SPARSE
)

# Check the description to get the host url
desc = pc.describe_index(name='sparse-index')

# Instantiate the index client
sparse_index = pc.Index(host=desc.host)
```

Upserting and querying is very similar to before, except now the `values` field of a Vector (used when working with dense values) may be unset.

```python
import random
from pinecone import Vector, SparseValues

def unique_random_integers(n, range_start, range_end):
    if n > (range_end - range_start + 1):
        raise ValueError("Range too small for the requested number of unique integers")
    return random.sample(range(range_start, range_end + 1), n)

# Generate some random sparse vectors
sparse_index.upsert(
    vectors=[
        Vector(
            id=str(i),
            sparse_values=SparseValues(
                indices=unique_random_integers(10, 0, 10000),
                values=[random.random() for j in range(10)]
            )
        ) for i in range(10000)
    ],
    batch_size=500,
)

# Querying sparse
sparse_index.query(
    top_k=10,
    sparse_vector={"indices":[1,2,3,4,5], "values": [random.random()]*5}
)
```

## Configuration UX with enums

Many enum objects have been added to help with the discoverability of some configuration options. Type hints in your editor will now suggest enums such as `Metric`, `AwsRegion`, `GcpRegion`, `PodType`, `EmbedModel`, `RerankModel` and more to help you quickly get going without having to go looking for documentation examples. This is a backwards compatible change and you should still be able to pass string values for fields exactly as before if you have preexisting code.

For example, code like this

```python
from pinecone import Pinecone, ServerlessIndex

pc = Pinecone()
pc.create_index(
    name='my-index',
    dimension=1536,
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-west-2'),
    vector_type='dense'
)
```

Can now be written as

```python
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    Metric,
    VectorType
)

pc = Pinecone()
pc.create_index(
    name='my-index',
    dimension=1536,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_WEST_2
    ),
    vector_type=VectorType.DENSE
)
```

Both ways of working are equally valid. Some may prefer the more concise nature of passing simple string values, but others may prefer the support your editor gives you to tab complete when working with enums.


# Upgrading from `4.x` to `5.x`

As part of an overall move to stop exposing generated code in the package's public interface, an obscure configuration property (`openapi_config`) was removed in favor of individual configuration options such as `proxy_url`, `proxy_headers`, and `ssl_ca_certs`. All of these properties were available in v3 and v4 releases of the SDK, with deprecation notices shown to affected users.

It is no longer necessary to install a separate plugin, `pinecone-plugin-inference`, to try out the [Inference API](https://docs.pinecone.io/guides/inference/understanding-inference); that plugin is now installed by default in the v5 SDK. See [usage instructions below](#inference-api).


# Upgrading from `3.x` to `4.x`

For this upgrade you are unlikely to be impacted by breaking changes unless you are using the `grpc` extras to use `PineconeGRPC` and have other dependencies in your project which place constraints on your grpc version. The `pinecone[grpc]` extras package got a breaking change to the underlying `grpcio` dependency to unlock significant performance improvements. Read full details in these [v4 Release Notes](https://github.com/pinecone-io/pinecone-python-client/releases/tag/v4.0.0).

# Upgrading to `3.x`:

Many things were changed in the v3 SDK to pave the way for Pinecone's new Serverless index offering as well as put in place a more object-oriented foundation for developing the SDK. These changes are covered in detail in the [**v3 Migration Guide**](https://canyon-quilt-082.notion.site/Pinecone-Python-SDK-v3-0-0-Migration-Guide-056d3897d7634bf7be399676a4757c7b#a21aff70b403416ba352fd30e300bce3). Serverless indexes are only available in `3.x` release versions or greater.
