# Reranking results

Reranking reorders candidate documents by relevance to a query. It runs a second, slower
model over candidates that something else already found (vector search, keyword search,
or another system), and usually buys precision at the top of the list at the cost of
latency.

## Basic usage

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

documents = [
    {"text": "Apple is a technology company."},
    {"text": "The apple is a popular fruit."},
    {"text": "Acme Inc. revolutionized enterprise software."},
]

result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="Tell me about tech companies",
    documents=documents,
)

for ranked in result.data:
    print(ranked.index, ranked.score, ranked.document)
```

A plain string is wrapped as `{"text": ...}`, which is the field the default
`rank_fields` scores on, so a list of strings works too:

```python
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="machine learning",
    documents=["Neural networks are a type of ML model.", "Python is a programming language."],
)
```

{meth}`~pinecone.client.inference.Inference.rerank` returns a
{class}`~pinecone.models.inference.rerank.RerankResult`:

- `.data`: the {class}`~pinecone.models.inference.rerank.RankedDocument` list, ordered by
  descending `score`.
- `.model`: the model that served the request. Pinecone can serve a request with a
  different model from the one you named, so read it here rather than assuming it
  echoes your argument.
- `.usage.rerank_units`: the rerank units counted for the call.

## Map results back with `index`

`result.data` is ordered by score, not by the order you passed the documents in. Each
`RankedDocument` carries the position it held in your `documents` argument as `.index`.
Reading `result.data` positionally and ignoring `.index` pairs your scores with the wrong
documents.

```python
top = result.data[0]
print(top.index)      # position in the documents you sent
print(top.score)      # relevance, higher is more relevant
print(top.document)   # the document as sent, or None with return_documents=False
```

To skip the bookkeeping, keep your own identifier on each document. Keys not named in
`rank_fields` come back unchanged in `.document`:

```python
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="Tell me about tech companies",
    documents=[
        {"id": "doc-1", "summary": "Apple is a fruit."},
        {"id": "doc-2", "summary": "Acme Inc. revolutionized tech."},
    ],
    rank_fields=["summary"],
)
print(result.data[0].document["id"])
```

## Choose what to score and return

- `rank_fields` names the document keys the model scores; it defaults to `["text"]`.
  Every document must have every named field, or the server rejects the request. Not
  every model accepts more than one rank field: `supported_parameters` on the model info
  says what a model takes (see
  [Discover what a model accepts](embeddings.md#discover-what-a-model-accepts)).
- `top_n` keeps only the best `n` results. By default every document comes back with a
  score.
- `return_documents=False` leaves `.document` as `None`, for when you already hold the
  documents and want only the ordering.

```python
result = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="quarterly earnings",
    documents=[
        {"title": "Acme Q4 results", "body": "Revenue grew year over year."},
        {"title": "Banana prices rise", "body": "Fruit prices hit new highs."},
    ],
    rank_fields=["title", "body"],
    top_n=1,
    return_documents=False,
)
print(result.data[0].index, result.data[0].score)
```

## Name a model

{class}`~pinecone.models.enums.RerankModel` gives tab-completion and typo safety in place
of a bare string. A model that is not an enum member still works as a string.

```python
from pinecone import Pinecone, RerankModel

pc = Pinecone(api_key="your-api-key")

result = pc.inference.rerank(
    model=RerankModel.Bge_Reranker_V2_M3,
    query="machine learning",
    documents=["Neural networks are a type of ML model."],
)
print(result.model)
```

A name the API does not serve raises {exc}`~pinecone.errors.exceptions.NotFoundError`.
A model your project is not authorized to use, including a deprecated one, raises
{exc}`~pinecone.errors.exceptions.ForbiddenError`. Check the name before suspecting the
request body.

List the reranking models available to your project:

```python
models = pc.inference.model.list(type="rerank")
print(models.names())
```

## Rerank search results

Two-stage retrieval fetches a wide candidate set from an index, then reranks it down.
This example searches the `product-search` index built in
[Storing embeddings in an index](embeddings.md#storing-embeddings-in-an-index):

```python
from pinecone import DenseVectorQuery, Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.index("product-search")
query_text = "shoes for muddy trails"

# Stage 1: vector search. Embed the query as a query, not a passage.
query_vector = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=query_text,
    parameters={"input_type": "query"},
).data[0].values

response = index.documents.search(
    namespace="products-en",
    top_k=20,
    score_by=[DenseVectorQuery(field="embedding", values=query_vector)],
    include_fields=["description"],
)

# Stage 2: rerank. Carry the document ID through so each score stays attributable.
if response.matches:
    reranked = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query_text,
        documents=[
            {"id": doc.id, "text": doc.get("description", "")}
            for doc in response.matches
        ],
        top_n=5,
    )
    for ranked in reranked.data:
        print(ranked.document["id"], ranked.score)
```

`top_k` decides how many candidates stage one retrieves and `top_n` how many survive
stage two. Reranking can only reorder what stage one found, so retrieve more than you
intend to keep. The SDK raises {exc}`~pinecone.errors.exceptions.PineconeValueError` for
an empty `documents` list, which is why the example checks `response.matches` first.

On a legacy integrated index, {meth}`~pinecone.index.Index.search` takes a `rerank`
argument and runs both stages in one request; see
[Rerank in a single search call](../integrated-records.md#rerank-in-a-single-search-call).
