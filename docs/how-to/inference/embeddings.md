# Generating embeddings

Pinecone hosts embedding models, so you can turn text into vectors without running
embedding infrastructure of your own. {meth}`~pinecone.client.inference.Inference.embed`
hands the vectors back to you, to upsert, to query with, or to store somewhere else.

If you would rather never handle a vector, give a document index a field with
integrated embedding, and Pinecone embeds on both write and read. See
[How Pinecone Works](../../guides/concepts.md#integrated-embedding). Legacy integrated
indexes do the same through the legacy Records API; see
[Legacy integrated indexes](../integrated-records.md).

## Basic usage

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["The quick brown fox", "A second piece of text"],
    parameters={"input_type": "passage"},
)

for embedding in embeddings:
    print(embedding.values[:3])
```

One embedding comes back per input, in input order. A bare string is wrapped for you, so
`inputs="one string"` still returns a one-item result rather than a lone embedding.

## `input_type`: the mismatch that costs you relevance

Many hosted models are asymmetric: they embed stored text and search queries
differently, and `input_type` says which side you are on. Embed your corpus as
`"passage"` and your queries as `"query"`:

```python
passages = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["Vector databases index embeddings for similarity search."],
    parameters={"input_type": "passage"},
)

query = pc.inference.embed(
    model="multilingual-e5-large",
    inputs="How does similarity search work?",
    parameters={"input_type": "query"},
)
```

Get this backwards and nothing fails: both calls succeed, both return vectors of the
right width, and search quality quietly degrades. Decide per call site which side you
are embedding.

## Discover what a model accepts

`parameters` is model-specific. Read the keys off the model rather than hardcoding a list
that will go stale:

```python
info = pc.inference.model.get("multilingual-e5-large")

for parameter in info.supported_parameters:
    print(parameter.parameter, parameter.allowed_values, parameter.default)
```

Each entry carries `allowed_values` when the set is fixed, `min`/`max` when the value is
numeric, `default` for what the model uses when you omit the key, and `required` for
whether you must send it.

The same {class}`~pinecone.models.inference.models.ModelInfo` reports the model's limits
and output shape:

```python
print(info.type)                # 'embed' or 'rerank'
print(info.vector_type)         # 'dense' or 'sparse'
print(info.default_dimension)   # output dimension when none is requested
print(info.max_batch_size)      # most inputs one call may carry
print(info.max_sequence_length) # longest input accepted
```

`vector_type` and `default_dimension` are `None` on a reranking model, and
`default_dimension` is `None` on a sparse model.

## Response: EmbeddingsList

`embed` returns an {class}`~pinecone.models.inference.embed.EmbeddingsList`. Iterating it
yields the embeddings, and `len()` and integer indexing reach the same items:

- `.data`: the same list the iteration walks, one entry per input.
- `.vector_type`: `"dense"` or `"sparse"`, and so which fields the entries carry.
- `.model`: the model that served the request.
- `.usage.total_tokens`: tokens counted for the call.

## Dense and sparse embeddings

The two shapes carry the vector in different fields. A
{class}`~pinecone.models.inference.embed.DenseEmbedding` has `values`; a
{class}`~pinecone.models.inference.embed.SparseEmbedding` has `sparse_values` and
`sparse_indices` paired position by position, and no `values` field. Branch on
`vector_type` when the model is not fixed in advance:

```python
first = embeddings.data[0]
if embeddings.vector_type == "sparse":
    print(first.sparse_indices, first.sparse_values)
else:
    print(first.values[:3])
```

Reading `.values` on a sparse embedding does not raise: it returns a dict-view method
instead of a vector, and your code carries on with the wrong object. The `vector_type`
check is what protects you.

One `embed` call returns either all-dense or all-sparse results, never a mix. For hybrid
search, call `embed` once with a dense model and once with a sparse model, and pair the
two result sets yourself.

## Using the EmbedModel enum

{class}`~pinecone.models.enums.EmbedModel` gives you tab-completion and typo safety in
place of a bare string:

```python
from pinecone import EmbedModel, Pinecone

pc = Pinecone(api_key="your-api-key")

embeddings = pc.inference.embed(
    model=EmbedModel.Multilingual_E5_Large,
    inputs=["search query"],
    parameters={"input_type": "query"},
)
```

`embed` accepts a plain string too, so a model that has no enum member yet still works.

## Batch size

Sending several inputs in one call saves round trips, up to the model's
`max_batch_size` (see [Discover what a model accepts](#discover-what-a-model-accepts)).
The SDK does not split an oversized batch; the server rejects it. Chunk your inputs
yourself:

```python
texts = [f"document number {n}" for n in range(250)]
batch_size = info.max_batch_size

all_embeddings = []
for start in range(0, len(texts), batch_size):
    batch = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=texts[start : start + batch_size],
        parameters={"input_type": "passage"},
    )
    all_embeddings.extend(batch.data)
```

Overlong individual inputs are a separate concern: where a model supports the `truncate`
parameter, it decides whether the model trims them or rejects the request.

## Storing embeddings in an index

Declare a `dense_vector` field at the model's output dimension, then upsert each
embedding's `values` into it:

```python
from pinecone import DenseVectorQuery, Pinecone

pc = Pinecone(api_key="your-api-key")
info = pc.inference.model.get("multilingual-e5-large")

pc.indexes.create(
    name="product-search",
    schema={
        "fields": {
            "embedding": {
                "type": "dense_vector",
                "dimension": info.default_dimension,
                "metric": "cosine",
            }
        }
    },
)
index = pc.index("product-search")

texts = ["Lightweight trail running shoes", "Waterproof hiking boots"]
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=texts,
    parameters={"input_type": "passage"},
)
index.documents.upsert(
    namespace="products-en",
    documents=[
        {"_id": f"prod-{i}", "embedding": embedding.values, "description": text}
        for i, (embedding, text) in enumerate(zip(embeddings, texts))
    ],
)
```

At query time, embed the query text with `input_type="query"` and score by the same
field:

```python
query = pc.inference.embed(
    model="multilingual-e5-large",
    inputs="shoes for muddy trails",
    parameters={"input_type": "query"},
)
results = index.documents.search(
    namespace="products-en",
    top_k=5,
    score_by=[DenseVectorQuery(field="embedding", values=query.data[0].values)],
)
for match in results.matches:
    print(match.id, match.score)
```

## List available models

```python
models = pc.inference.model.list(type="embed")
print(models.names())
```

`type` narrows the listing to `"embed"` or `"rerank"`; omit it for both. For embedding
models, `vector_type="dense"` or `"sparse"` narrows it further. Pairing `vector_type`
with `type="rerank"` raises {exc}`~pinecone.errors.exceptions.PineconeValueError`.

```python
sparse_models = pc.inference.model.list(type="embed", vector_type="sparse")
print(sparse_models.names())
```
