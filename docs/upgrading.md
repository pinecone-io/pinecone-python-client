# Upgrading from `5.x` to `6.x`

## Breaking changes in 6.x
- Dropped support for Python 3.8, which has now reached [official end of life](https://devguide.python.org/versions/). We added support for Python 3.13.
- Removed the explicit dependency on [`tqdm`](https://github.com/tqdm/tqdm) which is used to provide a nice progress bar when upserting lots of data into Pinecone. If `tqdm` is available in the environment the Pinecone SDK will detect and use it but we will no longer require `tqdm` to be installed in order to run the SDK. Popular notebook platforms such as [Jupyter](https://jupyter.org/) and [Google Colab](https://colab.google/) already include `tqdm` in the environment by default so for many users this will not require any changes, but if you are running small scripts in other environments and want to continue seeing the progress bars you will need to separately install the `tqdm` package.
- Removed some previously deprecated and rarely used keyword arguments `config` and `openapi_config` to instead prefer dedicated keyword arguments for individual settings such as `api_key`, `proxy_url`, etc. Having multiple ways of passing in the same configuration values was adding significant amounts of complexity to argument validation, testing, and documentation that wasn't really being repaid by significant ease of use, so we've removed those options.

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

    async with PineconeAsyncio().Index(host="")

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
